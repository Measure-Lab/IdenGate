# IDENGATE functional smoke-test report

## Test scope

This report records a code-path test of the release on a deterministic,
BreastMNIST-format binary dataset. It is a **functional smoke test**, not a
medical-image performance benchmark and not a replacement for training on the
official BreastMNIST file.

The test data use the MedMNIST NPZ keys and grayscale image format. Images are
loaded as 28×28 arrays and processed by the public pipeline to 224×224, then
repeated to three channels. The mini test uses 12 training, 6 validation, and 6
test images so that the full three-block model can be trained within the CPU
runtime available for this audit.

## Environment

```text
Python 3.13.5
PyTorch 2.10.0+cpu
Device: CPU
Input presented to model: 3×224×224
CSA grids: 56×56, 28×28, 14×14
Seed: 42
Epochs: 1
Batch size: 1
```

## Tests completed

| Test | Result |
|---|---|
| Python compilation | PASS |
| Unit tests | 7/7 PASS |
| Binary `--num-classes 2` model construction | PASS |
| Full-model training step at 224×224 | PASS |
| Validation macro-AUC checkpoint selection | PASS |
| Reload selected checkpoint with `weights_only=True` | PASS |
| Test evaluation only after checkpoint selection | PASS |
| no-MGF training path | PASS |
| Shuffle-MGF single-device training path | PASS |
| Full/Q/K/V/QK/QKV intervention pipeline | PASS |
| `alpha=0` versus all-role identity verification | PASS |
| Dataset SHA-256 mismatch rejection | PASS |

## Observed model sizes for the binary smoke test

```text
Full / Shuffle-MGF: 2,351,838 trainable parameters
no-MGF:              2,349,150 trainable parameters
```

The difference from the manuscript's RetinaMNIST counts is exactly due to the
classifier changing from five outputs to two outputs. The shared feature
extractor and gating architecture are unchanged.

## Interpretation

The smoke-test metrics are not scientifically meaningful because the images
are synthetic and the run uses one epoch. Their only role is to establish that
all training, checkpointing, loading, evaluation, intervention, calibration,
and audit paths execute without error for a binary MedMNIST-format task.

A performance claim requires the official BreastMNIST or RetinaMNIST data,
the manuscript training schedule, all five seeds, and the designated hardware
environment.
