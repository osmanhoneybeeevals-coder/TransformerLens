# Gemma 3 changes and fix summary

Draft for review — 26 October 2026

The follow-through adds clearer user guidance and independent regression coverage. CPU float32 checks passed without additional normalization or dtype changes. Recovering the original historical reproducer is no longer required for completing this scoped work. The implementation remains in a draft fork pull request; upstream publication is pending.

## The original issue and recorded fix

Historical triage and architecture notes describe an epsilon-placement error in normalization. Epsilon was applied outside the reciprocal square root calculation; the recorded correction moved it inside that calculation. Epsilon changes the normalization denominator, so its placement affects numerical outputs.

The notes report a longer-prompt maximum absolute logits error of about `1.3e-4` before the correction, exceeding the `1e-4` acceptance threshold. The recorded result after the fix was `2.9e-5`. These are historical reported measurements, not a before-and-after reproduction performed during this review.

The notes label the hotfix `v2.16.1` in June 2026, but that identity conflicts with the public package date in June 2025. This review therefore does not recommend that version as a verified installation target.

## What changed and why

### User guidance

Added [gemma3_validation.md](docs/source/content/gemma3_validation.md) and linked it from [the documentation index](docs/source/index.md). The guide covers version and dependency pinning, migration to TransformerBridge, numerical comparisons, verification-record limits, and vocabulary-size diagnostics.

This gives users actionable guidance without implying that every checkpoint, device or precision is certified. Raw logits and compatibility-mode probabilities have different comparison contracts, and a stored registry result does not establish a fresh pass on the user's environment.

### Regression coverage

Expanded [test_gemma3_adapter.py](tests/integration/model_bridge/test_gemma3_adapter.py) with independent Hugging Face comparisons at sequence lengths `1, 8, 9, 32`, around a local attention window of `8`.

The test builds the bridge from a deep copy of the HF model, keeping the HF reference independent. Bridge construction wraps modules in place; sharing the reference could hide a regression. The checks require finite outputs and explicitly set `atol=1e-4, rtol=0`, preventing a nonzero relative tolerance from weakening the absolute-error threshold.

### Completion criteria

Updated the review, guide and PR description to record the float32 outcome and remove recovery of the original historical reproducer as a prerequisite, as requested. Existing tests remain in place.

No new runtime, normalization or dtype fix was made in this follow-through. The changes improve guidance and regression protection around behavior that passed the current checks.

## How the changes were validated

The new checks compare:

- Raw bridge logits against independent HF logits.
- Compatibility-mode `log_softmax` against the HF reference.

Compatibility processing can shift raw logits while preserving probabilities, so these are distinct numerical contracts.

Environment: Python `3.12.14`, PyTorch `2.11.0+cu130`, Transformers `5.13.0`; a seeded tiny text model on CPU in float32.

| Input length | Raw maximum absolute error | Maximum log_softmax error |
| --- | --- | --- |
| 1 | 0 | 4.76837158203125e-7 |
| 8 | 0 | 4.76837158203125e-7 |
| 9 | 0 | 4.76837158203125e-7 |
| 32 | 0 | 9.5367431640625e-7 |

The targeted integration, adapter and RMSNorm-offset suite passed **41 tests**. Mypy passed across 389 source files; changed-file formatting and diff checks passed. The documentation scope update rebuilt successfully with five warnings, none naming the new guide.

The tests were unchanged by the later documentation-only scope update and by this Markdown summary. The 41-test result is from the implementation validation.

## Current status and limits

The tested CPU float32 behavior needs no further implementation change on the evidence reviewed. GPU, float16, bfloat16, pretrained checkpoints and multimodal inputs were not validated by these new tests. Untested configurations are not classified as broken. The full `test-pr` suite and exact-release CI were not run.

Upstream publication remains pending because the integration returned a 403 when creating the upstream PR. Merging the fork PR would not publish upstream documentation.

The separate vocabulary-size report remains unverified. A normalization fix does not establish that a loading or tensor-shape error is resolved.

## Implementation record and sources

- [Draft PR 1](https://github.com/osmanhoneybeeevals-coder/TransformerLens/pull/1)
- Branch: `docs/gemma3-followthrough`
- Implementation commit: `0ee660c01ab26f89905cab68fbf5c99e2d41f63b`
- Documentation scope update: `c3493c59ade9d6dfb5406558375eaf68e72b2793`
- Review base: upstream dev at `34c5d77f867be08442f6ed57b07d8e089beca6e2`
- [Public 2.16.1 package metadata](https://pypi.org/project/transformer-lens/2.16.1/)

Historical measurements were recorded in `triage_1121_gemma3_logits.md`, `gemma-3-arch.md`, and `release_notes_v2.16.1_2026-06-27.md`. Those internal source files were left unchanged. Private review and Drive links remain in the accompanying internal Word draft.

## Follow up repair for component verification

The current component-test path was repaired after reviewing upstream #1119 and #1120. Gemma 3 text and multimodal adapters now bind the correct local/global RoPE for each layer. The attention input generator supplies explicit causal/sliding-window masks and surfaces configured rotary errors instead of hiding them with identity rotations.

Ten new regression cases verify RoPE inputs, isolated attention parity and post-normalization block-hook ablations. The expanded targeted suite passed 498 tests. The block hooks matched HF interventions and needed no runtime forward change. See [the technical verification note](GEMMA3_TECHNICAL_VERIFICATION.md#component-rope-and-block-hook-repair) for the reproduced failures, commands and coverage limits. This repair changes component-test setup; it does not add a normalization or dtype fix.
