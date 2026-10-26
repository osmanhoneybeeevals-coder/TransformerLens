# Draft review note for PR 1119 and PR 1120

Draft for review — 26 October 2026

## Purpose and conclusion

This note records the review of the transformer-block, normalization and component-verification changes associated with upstream PRs #1119 and #1120. Both upstream PRs were already merged. The follow-up repair addresses problems reproduced on the current review branch; it does not reopen those PRs or claim that every historical change was defective.

The reproduced failures were in component-test input generation: missing layer-specific RoPE configuration and missing explicit attention masks. Those paths are now repaired on `docs/gemma3-followthrough`. Separate block-hook ablation checks matched independent Hugging Face interventions, so no BlockBridge forward change was needed.

## PR 1119 Gemma3 match

[Upstream PR #1119](https://github.com/TransformerLensOrg/TransformerLens/pull/1119) merged into `dev-3.x-folding` on November 9, 2025.

### What changed

- Gemma-3 post-attention and post-feedforward normalization mappings changed to RMSNorm bridges.
- Normalization delegated to the original HF implementation to preserve its computation and dtype behavior.
- Attention and component testing gained positional-embedding support and more consistent shared inputs.

### Why it mattered

Gemma-3 has normalization before and after its attention and MLP sublayers. Incorrect normalization behavior or mismatched positional inputs can produce disagreement with HF. Its local and global attention layers also need the appropriate RoPE configuration when tested independently.

The historical component-testing setup documented a shared-local-RoPE compromise. The current Transformers interface is different: its unified rotary module requires a layer-type argument. Reviewing the old PR motivated inspection of the current implementation, where that argument was missing.

## PR 1120 Gpt oss match

[Upstream PR #1120](https://github.com/TransformerLensOrg/TransformerLens/pull/1120) merged into `dev-3.x-folding` on November 11, 2025.

### What changed

The PR expanded beyond GPT-OSS integration into shared bridge behavior, including normalization folding, dtype handling, weight conversion and hook placement.

Its block changes recognized multiple normalization attribute names, including `ln_1`, `ln1` and `input_layernorm`. The block patch also removed a manual `hook_resid_mid` invocation because that hook was aliased to `ln2.hook_in`. The presence of this patch code alone does not prove that every execution path used it.

### Why it mattered

A hook must observe and modify the intended tensor at the intended stage. For Gemma-3, the attention and MLP contributions undergo post-normalization before being added to the residual stream. Capturing a tensor before that normalization would give a different intervention.

The PR discussion included a concern about changes unrelated to GPT-OSS; the author acknowledged the expanded scope. The review API returned no inline review threads for either PR. This is not evidence of a comprehensive independent review.

## What the current investigation reproduced

### Missing layer-specific RoPE

With Transformers 5.13.0, the component-test rotary call omitted `layer_type`. The input generator caught the resulting exception and substituted identity rotations.

New tiny text and multimodal tests both failed before the repair. In the reported cosine comparison, 120 of 128 elements differed from the HF reference; maximum absolute difference was `1.9899924993515015`.

### Missing attention masks

After repairing the rotary binding, isolated attention comparison exposed another mismatch. Component inputs used `attention_mask=None`, which gave standalone HF attention and bridge attention different causal semantics. Explicit, shared causal and sliding-window masks resolved that mismatch.

These observations concern the component-verification path. They do not establish a failure in ordinary full-model inference.

## What was fixed

Repair commit: [e637af3](https://github.com/osmanhoneybeeevals-coder/TransformerLens/commit/e637af3a78b43f2e13cbd0ed7d8515450d2225f0).

| Area | Repair |
| --- | --- |
| Gemma-3 text and multimodal adapters | Bind the rotary callable to each layer's attention type; use the first layer's type for the template. |
| Component attention inputs | Supply explicit causal masks and the wrapped layer's sliding-window restriction. |
| Rotary error handling | Propagate errors from configured rotary callables instead of silently substituting identity rotations. |
| Regression coverage | Add ten cases covering both model variants, local/global inputs, template setup, error propagation, attention parity and post-norm hook ablations. |

The unconfigured synthetic component retains its existing identity-rotation placeholder. The repair does not change runtime normalization or dtype handling.

## What was verified about the blocks

For tiny text and multimodal models, the new tests ablate the attention-output and MLP-output aliases and compare against independent HF interventions at the corresponding post-normalization modules.

Each tested alias fired once, observed the expected post-normalization value, and produced logits matching the equivalent HF intervention. No BlockBridge forward modification was needed.

These checks specifically cover the attention-output and MLP-output aliases. They are not a certification of every hook, layer, model architecture or intervention.

## Verification results

The expanded targeted suite passed **498 tests with 22 warnings in 120.95 seconds**. It covered the ten new regression cases, earlier Gemma-3 numerical parity tests, tiny multimodal image-path regressions, text/multimodal adapter unit tests, RMSNorm-offset tests and the generalized-component unit directory.

Environment: Python `3.12.14`, PyTorch `2.11.0+cu130`, Transformers `5.13.0`, CPU float32.

Mypy passed for 389 source files. Formatting completed and diff checks passed; the format recipe reported its known literal `isort format` path warning. The documentation build for the repair succeeded with 61 warnings.

The test count and build results describe the repair validation. This note is a documentation-only addition and does not change the test or runtime code.

## Remaining limits and publication status

- Full `test-pr`, pretrained-checkpoint, GPU, float16/bfloat16 and exact-release CI certification remain outside this verification.
- The current fix does not resolve the separate vocabulary-size/loading report.
- The historical hotfix artifact identity remains unverified. Neither upstream PR should be substituted for that artifact without evidence.
- Recovering the original historical reproducer is not a closure requirement for this scoped work.
- The repair is committed to the fork branch and included in [draft PR 1](https://github.com/osmanhoneybeeevals-coder/TransformerLens/pull/1). Upstream review and publication remain pending.

See [the change summary](GEMMA3_CHANGE_SUMMARY.md) for the overall follow-through and [the technical verification note](GEMMA3_TECHNICAL_VERIFICATION.md) for reproduction commands and detailed coverage.
