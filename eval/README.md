# IDENGATE: Paper-Aligned Core Implementation

This directory contains the publication-facing core implementation of:

> **IDENGATE: Bounded Role-Specific Preprojection Gating with Exact Identity Control for Medical Image Classification**

The code is organized around the central methodological claim of the paper: separate, bounded, image-specific gates are applied in the shared input-channel space before the query, key, and value projections, while the scaled dot-product attention operator remains unchanged. Setting the gate amplitude to zero restores exact identity gates in the same trained checkpoint.

This release is intentionally narrower and more auditable than the internal development code. It includes the primary model, the five-seed RetinaMNIST training protocol, fixed-checkpoint Q/K/V identity interventions, and automated integrity tests.

---

## Files

```text
IDENGATE_github_core/
├── model.py
├── train.py
├── identity_intervention.py
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── .gitignore
└── tests/
    └── test_model.py
```

### `model.py`

Implements:

- Adaptive Local Mapping Encoder (ALME);
- Modulated Gating Function (MGF);
- Conditioned Self-Attention (CSA);
- one-, two-, and three-block IDENGATE variants;
- exact MGF ON / MGF OFF behavior;
- Q/K/V role-specific identity interventions;
- parameter-matched Shuffle-MGF behavior;
- separately instantiated no-MGF control.

### `train.py`

Implements the paper-aligned five-seed RetinaMNIST protocol:

- official train, validation, and test splits;
- seeds 42–46;
- 60 epochs;
- batch size 128;
- AdamW with learning rate `3e-4` and weight decay `1e-4`;
- cosine annealing with warm restarts, `T_0=10`, `T_mult=2`;
- cross-entropy loss;
- no class weighting;
- no pretraining;
- validation macro-AUC checkpoint selection;
- test evaluation only after checkpoint selection.

### `identity_intervention.py`

Reproduces the fixed-checkpoint protocol underlying Table II:

- Full;
- Q identity;
- K identity;
- V identity;
- Q,K identity;
- Q,K,V identity;
- separate validation-fitted temperature for each state;
- 10-bin ECE;
- empirical risk–coverage and AURC;
- prediction-change rate relative to Full;
- seed-matched deltas and nominal paired 95% confidence intervals.

---

## 1. Exact method semantics

The following terms are used consistently in the paper and code.

| Term | Definition |
|---|---|
| **MGF ON** | Learned-gate state of a trained IDENGATE checkpoint, normally `alpha=0.1`. |
| **MGF OFF** | Exact identity-gate state of the same checkpoint, obtained with `alpha=0.0`. |
| **no-MGF** | A separately instantiated and separately trained model with no MGF parameters. |
| **Shuffle-MGF** | A parameter-matched control that jointly permutes the Q/K/V gate tuple across the mini-batch, preserving the gate-vector multiset while breaking image–gate correspondence. |
| **Q/K/V identity intervention** | Only the selected role gates are replaced by identity at inference. Other gates and all model computations remain fixed. |

MGF OFF is not a separately trained baseline. The implementation keeps ALME, the descriptor branch, learned projection weights, normalization statistics, input preprocessing, and all non-gating computations active.

---

## 2. Mathematical implementation

For tokenized ALME features

\[
X\in\mathbb{R}^{B\times L\times C},
\]

MGF produces role-specific unit-centered gates

\[
\phi^{(r)}(x;\alpha)
=
\mathbf{1}+\alpha\tanh(g_r(m(x))),
\qquad r\in\{Q,K,V\}.
\]

The gates act before the internal projection matrices of `torch.nn.MultiheadAttention`:

\[
Q=XD_QW_Q,\qquad
K=XD_KW_K,\qquad
V=XD_VW_V.
\]

The attention operation is unchanged:

\[
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V.
\]

At the primary amplitude,

\[
\alpha=0.1,
\]

every multiplier lies in `[0.9, 1.1]`. At `alpha=0.0`, all role gates are exactly one.

---

## 3. Reported parameter counts

The code verifies the manuscript's parameter counts:

| Configuration | Trainable parameters |
|---|---:|
| Full IDENGATE, 3 blocks, 5 classes | **2,353,377** |
| no-MGF control, 3 blocks, 5 classes | **2,350,689** |
| 1-block IDENGATE | approximately 0.12 M |
| 2-block IDENGATE | approximately 0.57 M |
| 3-block IDENGATE | approximately 2.35 M |

Run:

```bash
python model.py
```

Expected output:

```text
IDENGATE parameters: 2,353,377
no-MGF parameters:   2,350,689
```

---

## 4. Installation

### Conda

```bash
conda env create -f environment.yml
conda activate idengate
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The reference code targets Python 3.10 and PyTorch 2.8. Use the PyTorch wheel appropriate for the local CUDA driver.

---

## 5. Dataset preparation

Use the official RetinaMNIST NPZ file with these arrays:

```text
train_images
train_labels
val_images
val_labels
test_images
test_labels
```

The training code does not merge the validation split into training. This is required because the highest validation macro-AUC checkpoint is selected before test evaluation.

Example path:

```text
data/retinamnist_224.npz
```

The implementation resizes all inputs to `224 × 224`. The original implementation details retained in this release are:

- training random crop with padding 16;
- random horizontal flip;
- tensor normalization with mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`;
- one-channel images repeated to three channels.

---

## 6. Train the primary five-seed model

```bash
python train.py \
  --npz data/retinamnist_224.npz \
  --output-dir outputs/retinamnist_primary \
  --control full \
  --seeds 42 43 44 45 46
```

Defaults already match the manuscript:

```text
epochs          60
batch size      128
optimizer       AdamW
learning rate   3e-4
weight decay    1e-4
scheduler       CosineAnnealingWarmRestarts
T0              10
Tmult           2
loss            CrossEntropyLoss
selection       highest validation macro-AUC
class weighting none
pretraining     none
alpha           0.1
```

### Train the separately trained no-MGF control

```bash
python train.py \
  --npz data/retinamnist_224.npz \
  --output-dir outputs/retinamnist_no_mgf \
  --control no_mgf \
  --seeds 42 43 44 45 46
```

### Train the parameter-matched Shuffle-MGF control

```bash
python train.py \
  --npz data/retinamnist_224.npz \
  --output-dir outputs/retinamnist_shuffle_mgf \
  --control shuffle_mgf \
  --seeds 42 43 44 45 46
```

The Shuffle-MGF implementation applies one batch permutation jointly to the Q/K/V gate tuple inside each block. It preserves the mini-batch gate-vector multiset and breaks the correspondence between each image and its gates.

---

## 7. Training outputs

For each seed:

```text
outputs/retinamnist_primary/
├── configuration.json
├── seed_42/
│   ├── best_validation_macro_auc.pt
│   ├── history.csv
│   └── test_metrics.json
├── seed_43/
├── seed_44/
├── seed_45/
├── seed_46/
├── seed_results.csv
└── summary.json
```

Each checkpoint contains:

- model state dictionary;
- complete model configuration;
- complete training configuration;
- control type;
- random seed;
- selected epoch;
- validation-selection metric;
- validation metrics.

The summary records:

- seed-level test accuracy and macro-AUC;
- mean, standard deviation, and 95% interval;
- checkpoint SHA-256 hashes;
- NPZ SHA-256 hash;
- Python, PyTorch, CUDA, cuDNN, platform, and GPU metadata.

---

## 8. Reproduce fixed-checkpoint identity interventions

After training the five Full checkpoints:

```bash
python identity_intervention.py \
  --npz data/retinamnist_224.npz \
  --checkpoint-root outputs/retinamnist_primary \
  --output-dir outputs/identity_interventions \
  --seeds 42 43 44 45 46
```

The script evaluates:

```text
full
q_identity
k_identity
v_identity
qk_identity
qkv_identity
```

For every state:

1. the same trained checkpoint is loaded;
2. the same official validation and test splits are used;
3. normalization statistics remain fixed;
4. ALME and descriptor computations remain active;
5. only selected role gates are set to identity;
6. a scalar temperature is fitted to that state's validation logits;
7. ECE is computed with 10 equal-width bins on test probabilities;
8. AURC uses empirical risk–coverage from raw maximum-softmax confidence;
9. prediction changes are measured relative to Full;
10. seed-matched deltas use Student's `t` interval for five seeds.

The script also verifies numerically that:

```text
Q,K,V identity at alpha=0.1 == alpha=0 in the same checkpoint
```

with zero absolute and relative tolerance.

Expected output:

```text
outputs/identity_interventions/
├── seed_42.json
├── seed_43.json
├── seed_44.json
├── seed_45.json
├── seed_46.json
├── table_ii_reproduced.csv
└── summary.json
```

---

## 9. Direct model API

```python
import torch
from model import build_primary_model

model = build_primary_model().eval()
x = torch.randn(2, 3, 224, 224)

# Primary learned-gate state: MGF ON
logits_on = model(x, alpha=0.1)

# Exact same-checkpoint identity state: MGF OFF
logits_off = model(x, alpha=0.0)

# Role-specific interventions
logits_q = model(x, alpha=0.1, identity_roles=("Q",))
logits_k = model(x, alpha=0.1, identity_roles=("K",))
logits_v = model(x, alpha=0.1, identity_roles=("V",))
logits_qk = model(x, alpha=0.1, identity_roles=("Q", "K"))
logits_qkv = model(x, alpha=0.1, identity_roles=("Q", "K", "V"))

# Return role gates for auditing
logits, stage_gates = model(x, alpha=0.1, return_gates=True)
```

Do not use `use_mgf=False` to represent MGF OFF. `use_mgf=False` defines the separately trained no-MGF architecture.

---

## 10. Integrity tests

Run:

```bash
pytest -q
```

The included tests verify:

- exact Full and no-MGF parameter counts;
- exact equality between `alpha=0` and all-role identity intervention;
- `[0.9, 1.1]` gate bounds at `alpha=0.1`;
- non-selected gates remain unchanged during a role intervention;
- Shuffle-MGF preserves the gate-vector multiset.

Expected result:

```text
5 passed
```

---

## 11. Paper-to-code mapping

| Paper element | Implementation |
|---|---|
| ALME, Eq. 6 | `model.py::ALME` |
| MGF, Eqs. 7–9 | `model.py::ModulatedGatingFunction` |
| Q/K/V preprojection gating, Eqs. 1–5 | `model.py::ConditionedSelfAttention` |
| Three-block network | `model.py::IDENGATE` |
| MGF ON / MGF OFF | `IDENGATE.forward(..., alpha=0.1/0.0)` |
| Q/K/V interventions | `identity_roles` argument and `identity_intervention.py` |
| no-MGF control | `train.py --control no_mgf` |
| Shuffle-MGF control | `train.py --control shuffle_mgf` |
| Five-seed protocol | `train.py` defaults and `PAPER_SEEDS` |
| Table II metrics | `identity_intervention.py` |

---

## 12. Reproducibility rules

The following invariants must not be changed in paper reproduction runs:

1. retain the official train, validation, and test split separation;
2. select checkpoints by validation macro-AUC, never test accuracy;
3. use seeds 42–46 for the five-seed mechanism study;
4. keep `alpha=0.1` for the primary learned-gate state;
5. implement MGF OFF with `alpha=0` in the same checkpoint;
6. keep ALME and descriptor computations active for MGF OFF;
7. set only selected role gates to identity during interventions;
8. fit temperature on validation logits separately for each state;
9. compute ECE using 10 equal-width bins;
10. preserve seed-level results before aggregation;
11. retain checkpoint and data hashes;
12. do not describe source-reported external baselines as commonly retrained comparisons.

---

## 13. Notes on numerical reproducibility

GPU kernels, CUDA versions, and mixed-precision execution can produce small numerical differences. Reproduction should preserve:

- experimental protocol;
- split assignments;
- seed matching;
- checkpoint-selection criterion;
- intervention semantics;
- metric definitions;
- aggregation method.

The reported result should not be reconstructed by manually inserting target values into generated files.

---

## 14. Intended use

The implementation is provided for research and reproducibility. It is not a certified medical device and must not be used for diagnosis, patient management, or clinical deployment without independent validation, institutional approval, and applicable regulatory review.

---

## 15. Citation

```bibtex
@misc{huang2026idengate,
  title  = {IDENGATE: Bounded Role-Specific Preprojection Gating with Exact Identity Control for Medical Image Classification},
  author = {Hao Huang and Jee-Hyong Lee and Ce Gao and Ning Wang and Zongrun Sun},
  year   = {2026},
  note   = {Manuscript submitted to IEEE Transactions on Medical Imaging}
}
```
