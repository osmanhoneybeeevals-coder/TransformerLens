# Gemma 3 version pinning and numerical validation

Use `TransformerBridge` for new Gemma 3 work. When moving an existing experiment
to TransformerLens 4, follow the [migration guide](migrating_to_v4.md):
`HookedTransformer.from_pretrained` is no longer available. Validate the new
environment before replacing results from an older one.

## Which version should I pin?

Pin the exact release or source commit that passed your workload, together with
its resolved dependencies. A release announcement or an adapter's presence is
not a production qualification for every checkpoint, device, or precision.
Use the [release history](https://github.com/TransformerLensOrg/TransformerLens/releases)
to identify a fix and check that the installed artifact contains it. Do not infer
that an old version is the recommended pin from an issue number alone.

For a paper, retain the environment that produced the reported results. Record:

- TransformerLens version and source commit for a source installation;
- Python, PyTorch, Transformers and the dependency lockfile;
- exact model and tokenizer repository IDs and immutable revisions;
- device, dtype, attention implementation, seed and input token IDs;
- compatibility-mode settings and any hooks or weight edits;
- the comparison metric, absolute and relative tolerances, and measured errors.

Repeat the affected experiments after a numerical fix and identify corrected
results. Do not silently replace a paper's environment with the newest release.

## Compare the right numerical contract

| Configuration | Comparison | Why |
| --- | --- | --- |
| Raw bridge, no hooks or weight processing | Raw logits against an independent HF model | Checks the native forward contract. |
| Compatibility mode with unembedding centering | `log_softmax` against the raw HF reference | Centering may shift every vocabulary logit by a common value while preserving probabilities. |
| Hooked or edited execution | The intended intervention, with an unedited control | A deliberate intervention need not preserve the original logits. |

Use identical weights, input IDs, masks, positional inputs, dtype, device and
attention implementation. Set both models to evaluation mode. Keep the HF
reference independent: building a bridge wraps modules in place, so reusing the
same HF object as the oracle can hide a regression. See
[debugging numerical divergence](debugging_numerical_divergence.md) and
[compatibility mode](compatibility_mode.md).

Check finite values and tensor shapes before calculating errors. Report maximum
absolute error and set both `atol` and `rtol` explicitly; `atol=1e-4` alone is not
a strict maximum-error guarantee when a nonzero relative tolerance also applies.
Investigate errors before changing a threshold.

The offline Gemma 3 integration test uses a seeded tiny text model on CPU in
float32. It compares independent HF raw logits and compatibility-mode
`log_softmax` at sequence lengths 1, 8, 9 and 32 with a local window of 8,
using `atol=1e-4, rtol=0`. Run it from a contributor checkout:

```bash
uv run pytest tests/integration/model_bridge/test_gemma3_adapter.py
```

This test does not certify pretrained checkpoints, GPU/reduced-precision runs,
multimodal input, or a historical bug report's reproducer. Check the exact
checkpoint's entry in the [model table](model_tables.md), including individual
phase results and notes. A missing phase result is not a passing result, and a
registry record is not a fresh run on your installed release.

## Vocabulary-size and loading errors

A normalization fix does not establish that a vocabulary-size mismatch is fixed.
First record the exact exception and the tensor names and shapes in it. Compare
the checkpoint, tokenizer and configuration revisions; distinguish the text
model configuration from a multimodal wrapper's configuration.

Record tokenizer length, maximum token ID, text-config vocabulary size, input
embedding rows and output-head rows. These need interpretation: padded embedding
tables and added tokens can make counts differ legitimately. Verify that the
actual input IDs are in range and that checkpoint tensor shapes match the model
being loaded. Do not truncate weights, force a vocabulary size, or suppress
mismatched-weight errors just to make loading succeed.

Include this information in a minimal issue report:

```text
TransformerLens version or commit:
Python / torch / transformers versions:
Model ID and immutable revision:
Tokenizer ID and immutable revision:
Model class and text-only or multimodal input:
Device / dtype / attention implementation:
Compatibility-mode settings:
Tokenizer length / maximum token ID:
Text-config vocabulary size:
Input embedding / output-head shapes:
Full traceback and minimal loading code:
```

For a logits mismatch, also include exact input tokens and masks, the independent
HF comparison, measured errors and tolerances. For a claimed regression, retain
the same reproducer on the failing and fixed artifacts. A successful tiny test
does not close a separate user loading failure.
