# Gemma 3 technical verification note

Verification date: 26 October 2026  
Status: scoped CPU float32 verification; draft for review

## Purpose and code identity

This note records the verification of the documentation and regression-test follow-through in [draft PR 1](https://github.com/osmanhoneybeeevals-coder/TransformerLens/pull/1). It supplements [the change summary](GEMMA3_CHANGE_SUMMARY.md) and [the user guide](docs/source/content/gemma3_validation.md).

- Branch: `docs/gemma3-followthrough`.
- Upstream dev baseline: `34c5d77f867be08442f6ed57b07d8e089beca6e2`.
- Test implementation commit: `0ee660c01ab26f89905cab68fbf5c99e2d41f63b`.
- Documentation scope update: `c3493c59ade9d6dfb5406558375eaf68e72b2793`.
- Branch head before this note: `2f16776f3ad52393876853166fca64c42acd8910`.

The follow-through changes documentation and tests. It does not modify runtime normalization or dtype handling. Historical notes report an epsilon-placement correction, but this verification does not recreate that historical failure or establish the exact historical release artifact.

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
