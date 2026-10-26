# Gemma 3 technical verification note

Verification date: 26 October 2026  
Status: CPU float32 verification plus component RoPE and block-hook regression coverage; draft for review

## Purpose and code identity

This note records the verification of the documentation and regression-test follow-through in [draft PR 1](https://github.com/osmanhoneybeeevals-coder/TransformerLens/pull/1). It supplements [the change summary](GEMMA3_CHANGE_SUMMARY.md) and [the user guide](docs/source/content/gemma3_validation.md).

- Branch: `docs/gemma3-followthrough`.
- Upstream dev baseline: `34c5d77f867be08442f6ed57b07d8e089beca6e2`.
- Test implementation commit: `0ee660c01ab26f89905cab68fbf5c99e2d41f63b`.
- Documentation scope update: `c3493c59ade9d6dfb5406558375eaf68e72b2793`.
- Branch head before this note: `2f16776f3ad52393876853166fca64c42acd8910`.

The initial follow-through changed documentation and tests. The later component-test repair below also changes test-input generation and per-layer RoPE wiring. Neither change modifies runtime normalization or dtype handling. Historical notes report an epsilon-placement correction, but this verification does not recreate that historical failure or establish the exact historical release artifact.

## Environment and fixture

| Setting | Verified configuration |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.13.0 |
| Execution | CPU, float32, eager attention |
| Model | Randomly initialized Gemma3ForCausalLM with Gemma3TextConfig |
| Vocabulary and hidden size | 200 and 64 |
| Intermediate size and layers | 128 and 4 |
| Attention heads and KV heads | 4 and 2 |
| Head dimension | 16 |
| Local window and maximum positions | 8 and 256 |
| Weight seed and token seed | 42 and 7 |
| Input shape | Batch size 1; lengths 1, 8, 9 and 32 |
| Input IDs | Random integers from 3 through 199 |

The fixture uses evaluation mode and `torch.no_grad()`. It creates no downloaded checkpoint and uses no tokenizer. The CUDA suffix in the PyTorch version does not imply GPU execution.

## Independence and assertions

The test retains the original HF model as its reference and builds the bridge from `copy.deepcopy(hf)`. This matters because bridge construction wraps modules in place: wrapping the same object used as the reference could conceal a regression.

For each input length, the test checks finite HF and bridge outputs, then applies:

1. **Raw mode:** `torch.testing.assert_close(raw, expected, atol=1e-4, rtol=0)`.
2. **Compatibility mode:** enable compatibility mode and compare `processed.float().log_softmax(dim=-1)` against the equivalent HF reference, with the same tolerances.

With zero relative tolerance, the comparison requires absolute error no greater than `1e-4` for each element, in addition to the shape and dtype checks made by `assert_close`. Compatibility mode can shift vocabulary logits by a common value, so raw-logit equality is not the compatibility-mode contract.

The selected lengths exercise a short input, the local-window boundary, the first length beyond it, and a longer input. They do not establish arbitrary-length or generation-cache correctness.

## Measured numerical results

The following maximum errors were measured during the original implementation validation:

| Input length | Maximum absolute raw-logit error | Maximum absolute log_softmax error |
| --- | --- | --- |
| 1 | 0 | 4.76837158203125e-7 |
| 8 | 0 | 4.76837158203125e-7 |
| 9 | 0 | 4.76837158203125e-7 |
| 32 | 0 | 9.5367431640625e-7 |

Maximum absolute error means `max(abs(actual - expected))` over all output elements. These observed values are specific to this fixture and environment, not universal error bounds.

## Reproduction commands

Use a contributor checkout containing the implementation commit, with the dependencies above already installed. Record versions before interpreting a run; a newly resolved environment may differ.

```bash
uv run --no-sync python -c 'import sys, torch, transformers; print(sys.version); print(torch.__version__); print(transformers.__version__)'

uv run --no-sync pytest \
  tests/integration/model_bridge/test_gemma3_adapter.py \
  tests/unit/model_bridge/supported_architectures/test_gemma3_adapter.py \
  tests/unit/model_bridge/supported_architectures/test_gemma_rmsnorm_offset.py -q
```

To run only the four new independent-reference cases:

```bash
uv run --no-sync pytest \
  tests/integration/model_bridge/test_gemma3_adapter.py::test_gemma3_raw_and_processed_against_independent_hf -q
```

`--no-sync` preserves an existing environment; it does not install or lock dependencies. These tests construct tiny models locally and do not require Hugging Face downloads. The pytest command verifies assertions; it does not print the numerical-error table above.

## Verification outcome and interpretation

A fresh run on 26 October 2026 completed with **41 passed, 2 warnings in 12.16 seconds** in the environment above. The warnings were reported by pytest; they did not fail the run. The numerical-error table is retained from the earlier measurement run, not newly emitted by this pytest invocation.

The targeted suite covers Gemma 3 integration behavior, adapter unit behavior and RMSNorm-offset unit behavior. The 41-test count describes the combined suite; the new parameterized independent-HF test contributes four cases.

The result supports the tested CPU float32 contracts without further normalization or dtype changes. Recovering the original historical reproducer is not a requirement for this scoped follow-through.

## Limits and outstanding work

- No pretrained-checkpoint, GPU, float16, bfloat16, multimodal or generation-cache certification is established by this fixture.
- Random single-batch token inputs do not validate tokenizer behavior, padded batches or a user's production workload.
- Full `test-pr` and exact-release CI were not run for this work.
- The historical release identity remains unresolved; the reported June 2026 hotfix label must not be treated as a verified public version pin.
- The separate vocabulary-size/loading report is not shown to be resolved by these numerical tests.
- The fork PR remains a draft. Upstream documentation publication is pending; upstream PR creation previously returned 403.

Untested configurations are not classified as broken. Historical source records remain unchanged.

## Component RoPE and block hook repair

Review of upstream PRs #1119 and #1120 prompted a fresh check of the current component-test path. This repair is against the current branch baseline, not a modification of those already merged PRs.

### Reproduced failures

With Transformers 5.13.0, Gemma 3 rotary embedding requires a layer type. The existing shared rotary callable omitted it; the component input generator caught the resulting error and substituted identity rotations. The new text and multimodal tests both failed before the fix: 120 of 128 cosine elements differed from HF, with maximum absolute difference 1.9899924993515015.

After repairing the rotary binding, direct attention comparison exposed a second mismatch: the input generator supplied no mask, so standalone HF attention and the bridge used different causal semantics. Supplying the same explicit mask resolved that mismatch.

### Implementation

- Both Gemma 3 adapters bind the rotary callable separately for each layer type. The template uses the first layer's type; re-running setup is safe. The helper also selects separate local/global rotary modules when that older HF layout is present.
- PositionEmbeddingsAttentionBridge generates an explicit causal mask and applies the wrapped layer's sliding-window limit where present. This affects component-test input generation, not the normal forward path.
- Errors from a configured rotary callable now propagate instead of silently falling back to identity rotations. An unconfigured synthetic component still retains its existing identity placeholder.
- No BlockBridge forward modification was needed. New ablation checks verify that attention-output and MLP-output aliases fire once after post-normalization and match equivalent independent HF interventions.

### New regression coverage

`tests/integration/model_bridge/test_gemma3_component_rope.py` contributes ten cases across tiny text and multimodal models: local/global RoPE values and attention-output parity, template wiring, explicit propagation of rotary failures, and post-norm attention/MLP ablations. Each model has one sliding-attention and one full-attention layer, with sequence length 16 for the component checks and window 8. Tests assert that future keys are masked, distant keys are masked only for the local layer, and local/global rotations differ.

The expanded run passed **498 tests with 22 warnings in 120.95 seconds**. It included the ten new cases, the earlier Gemma 3 parity tests, tiny multimodal image-path regressions, text/multimodal adapter unit tests, RMSNorm-offset tests and the generalized-component unit directory. Mypy passed for 389 source files. This does not establish pretrained, GPU, reduced-precision or full-suite certification.

Reproduce the expanded run:

```bash
uv run --no-sync pytest \
  tests/integration/model_bridge/test_gemma3_component_rope.py \
  tests/integration/model_bridge/test_gemma3_adapter.py \
  tests/integration/model_bridge/test_gemma3_multimodal_tiny.py \
  tests/unit/model_bridge/supported_architectures/test_gemma3_adapter.py \
  tests/unit/model_bridge/supported_architectures/test_gemma3_multimodal_adapter.py \
  tests/unit/model_bridge/supported_architectures/test_gemma_rmsnorm_offset.py \
  tests/unit/model_bridge/generalized_components -q
```

The environment remains Python 3.12.14, PyTorch 2.11.0+cu130 and Transformers 5.13.0, on CPU. Historical 41-test and numerical-error results above describe the earlier phase; the 498-test result describes this repair.
