"""Offline component checks for both Gemma 3 attention layer types."""

import copy

import pytest
import torch
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    SiglipVisionConfig,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture(params=[False, True], ids=["text", "multimodal"])
def models(request):
    multimodal = request.param
    text = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
    )
    if multimodal:
        vision = SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        )
        config = Gemma3Config(
            text_config=text.to_dict(),
            vision_config=vision.to_dict(),
            mm_tokens_per_image=4,
            image_token_index=31,
        )
        hf = Gemma3ForConditionalGeneration(config).eval()
        lm = hf.model.language_model
        architecture = "Gemma3ForConditionalGeneration"
    else:
        config = text
        hf = Gemma3ForCausalLM(config).eval()
        lm = hf.model
        architecture = "Gemma3ForCausalLM"
    bridge = build_bridge_from_module(
        copy.deepcopy(hf),
        architecture=architecture,
        hf_config=copy.deepcopy(config),
        device="cpu",
    ).eval()
    bridge.adapter.setup_component_testing(hf, bridge_model=bridge)
    return text, hf, lm, bridge


def test_component_inputs_use_each_layers_rope(models):
    text, hf, lm, bridge = models
    # Rewiring must remain safe and retain layer-specific inputs.
    bridge.adapter.setup_component_testing(hf, bridge_model=bridge)
    actual_cos = []
    for index, layer_type in enumerate(text.layer_types):
        attention = bridge.blocks[index].attn
        inputs = attention.get_random_inputs(batch_size=1, seq_len=16)
        positions = torch.arange(16).unsqueeze(0)
        expected = lm.rotary_emb(inputs["hidden_states"], positions, layer_type=layer_type)
        for actual, reference in zip(inputs["position_embeddings"], expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        actual_cos.append(inputs["position_embeddings"][0])
        mask = inputs["attention_mask"][0, 0]
        assert mask[0, 1] < -1e30
        assert mask[15, 15] == 0
        assert (mask[15, 0] < -1e30).item() == (layer_type == "sliding_attention")
        with torch.no_grad():
            reference_output = lm.layers[index].self_attn(**inputs)[0]
            actual_output = attention(**inputs)[0]
        torch.testing.assert_close(actual_output, reference_output, atol=1e-6, rtol=1e-5)
    assert not torch.equal(actual_cos[0], actual_cos[1])


def test_component_template_uses_first_layers_rope(models):
    text, hf, lm, bridge = models
    bridge.adapter.setup_component_testing(hf)
    template = bridge.adapter.get_generalized_component("blocks.0.attn")
    inputs = template.get_random_inputs(batch_size=1, seq_len=16)
    expected = lm.rotary_emb(
        inputs["hidden_states"], torch.arange(16).unsqueeze(0), layer_type=text.layer_types[0]
    )
    for actual, reference in zip(inputs["position_embeddings"], expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_configured_rotary_failure_is_not_silently_replaced(models):
    _, _, _, bridge = models

    def broken_rotary(*args, **kwargs):
        raise ValueError("invalid rotary configuration")

    bridge.blocks[0].attn.set_rotary_emb(broken_rotary)
    with pytest.raises(ValueError, match="invalid rotary configuration"):
        bridge.blocks[0].attn.get_random_inputs()


@pytest.mark.parametrize(
    "alias,norm_name",
    [("hook_attn_out", "post_attention_layernorm"), ("hook_mlp_out", "post_feedforward_layernorm")],
)
def test_block_output_ablation_matches_hf_post_norm(models, alias, norm_name):
    _, hf, lm, bridge = models
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11]])
    observed = []

    def ablate(value, hook):
        observed.append(value.detach().clone())
        return torch.zeros_like(value)

    reference_values = []

    def hf_ablate(module, args, output):
        reference_values.append(output.detach().clone())
        return torch.zeros_like(output)

    handle = getattr(lm.layers[0], norm_name).register_forward_hook(hf_ablate)
    try:
        with torch.no_grad():
            expected = hf(tokens).logits
            with bridge.hooks(fwd_hooks=[(f"blocks.0.{alias}", ablate)]):
                actual = bridge(tokens)
    finally:
        handle.remove()
    assert len(observed) == len(reference_values) == 1
    torch.testing.assert_close(observed[0], reference_values[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
