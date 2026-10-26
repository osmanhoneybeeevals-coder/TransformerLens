"""Integration tests for the Gemma3 adapter on a seeded tiny (no downloads).

Pins the compatibility-mode contract: processed weights must preserve the
predictive distribution (log_softmax parity; raw logits shift by design under
center_unembed).
"""

import copy

import pytest
import torch

pytest.importorskip("transformers", reason="requires transformers")

from transformers import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def _tiny_gemma3_hf():
    cfg = Gemma3TextConfig(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        sliding_window=8,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(42)
    return Gemma3ForCausalLM(cfg).eval()


def _tiny_gemma3_bridge():
    hf = _tiny_gemma3_hf()
    return build_bridge_from_module(
        hf, "Gemma3ForCausalLM", hf_config=copy.deepcopy(hf.config), tokenizer=None, device="cpu"
    ).eval()


@pytest.mark.parametrize("seq_len", [1, 8, 9, 32])
def test_gemma3_raw_and_processed_against_independent_hf(seq_len: int) -> None:
    """Check both numerical contracts across the local-attention boundary.

    The HF oracle is independent: bridge construction wraps modules in place.
    This tiny fp32 CPU test does not certify downloaded checkpoints or bf16.
    """
    hf = _tiny_gemma3_hf()
    bridge = build_bridge_from_module(
        copy.deepcopy(hf),
        "Gemma3ForCausalLM",
        hf_config=copy.deepcopy(hf.config),
        tokenizer=None,
        device="cpu",
    ).eval()
    torch.manual_seed(7)
    tokens = torch.randint(3, hf.config.vocab_size, (1, seq_len))
    with torch.no_grad():
        expected = hf(tokens).logits
        raw = bridge(tokens)
    assert torch.isfinite(expected).all()
    assert torch.isfinite(raw).all()
    torch.testing.assert_close(raw, expected, atol=1e-4, rtol=0)

    bridge.enable_compatibility_mode()
    with torch.no_grad():
        processed = bridge(tokens)
    assert torch.isfinite(processed).all()
    torch.testing.assert_close(
        processed.float().log_softmax(dim=-1),
        expected.float().log_softmax(dim=-1),
        atol=1e-4,
        rtol=0,
    )


class TestGemma3CompatibilityMode:
    def test_act_fn_is_tanh_gelu(self) -> None:
        bridge = _tiny_gemma3_bridge()
        assert bridge.cfg.act_fn == "gelu_pytorch_tanh"

    def test_compat_mode_preserves_log_softmax(self) -> None:
        """Seq length deliberately exceeds sliding_window so both mask paths run."""
        bridge = _tiny_gemma3_bridge()
        torch.manual_seed(7)
        tokens = torch.randint(3, 200, (1, 12))
        with torch.no_grad():
            base = bridge(tokens)
        bridge.enable_compatibility_mode()
        with torch.no_grad():
            proc = bridge(tokens)
        ls_base = torch.log_softmax(base.float(), dim=-1)
        ls_proc = torch.log_softmax(proc.float(), dim=-1)
        assert torch.allclose(
            ls_base, ls_proc, atol=1e-4
        ), f"log_softmax diverged: max={(ls_base - ls_proc).abs().max().item()}"
