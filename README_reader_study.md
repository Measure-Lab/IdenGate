# Exploratory Reader Study for IDENGATE

## Study design, de-identified data specification, statistical analysis, and figure reproduction

This directory documents the exploratory retrospective reader study reported in:

> **IDENGATE: Bounded Role-Specific Preprojection Gating with Exact Identity Control for Medical Image Classification**

The study evaluates whether IDENGATE-assisted evidence changes clinician confidence, decision time, confidence–accuracy calibration, and paired decision outcomes on RetinaMNIST images.

This component is intended for transparent analysis and reproducibility. It does **not** constitute prospective clinical validation, a diagnostic-performance claim, or evidence that the system is ready for clinical deployment.

---

## 1. Ethics and governance

The retrospective reader study was approved by the **Sungkyunkwan University Institutional Review Board**:

> **SKKU IRB No. 2026-01-011**

Both raters provided informed consent. The study used de-identified public images and did not affect patient care.

Only de-identified study records should be distributed. Names, institutional identifiers, contact information, free-text notes that may identify a rater, and any non-public source documents must not be included in the public repository.

Before publicly releasing row-level records, confirm that the approved protocol and informed-consent materials permit the intended form of data sharing. When row-level release is not permitted, publish only the analysis code, integrity checks, aggregate summaries, and the reproduced figure.

---

## 2. Recommended repository placement

Place this file at:

```text
IDENGATE/
└── reader_study/
    └── README.md
```

The figure is shared with the main repository README and remains in the existing `assets/` directory:

```text
IDENGATE/
├── README.md
├── assets/
│   └── reader_study.png
└── reader_study/
    └── README.md
```

Accordingly, this README uses the relative path:

```markdown
../assets/reader_study.png
```

Do not create a second copy of the same figure inside `reader_study/`.

---

## 3. Study overview

### 3.1 Design

The analysis contains:

- **2 raters**;
- **100 unique RetinaMNIST images**;
- **2 conditions** per image and rater;
- **200 matched image–rater pairs**;
- **400 total readings**.

Each rater evaluated the same 100 images under:

1. **Unaided condition** (`No-assist`);
2. **IDENGATE-assisted condition** (`AI-assist`).

For every image–rater pair, the dataset contains exactly one unaided reading and one assisted reading. The primary comparisons are therefore paired within image and rater.

### 3.2 Scope

The study assesses:

- change in clinician confidence;
- change in decision time;
- confidence–accuracy calibration;
- paired changes in correctness;
- descriptive automation-risk patterns among high-confidence AI-assisted reads.

The study is explicitly exploratory. It was not designed to establish clinical non-inferiority, clinical efficacy, diagnostic safety, or deployment readiness.

---

## 4. Summary figure

<p align="center">
  <img src="../assets/reader_study.png" width="960" alt="Exploratory IDENGATE reader-study results">
</p>

The figure contains four panels:

- **Panel a:** paired confidence change for each rater and overall;
- **Panel b:** paired decision-time change for each rater and overall;
- **Panel c:** 10-bin clinician confidence–accuracy calibration;
- **Panel d:** descriptive automation-risk audit at selected AI top-1 probability thresholds.

---

## 5. Primary results

### 5.1 Confidence change

Confidence change is defined as:

\[
\Delta \mathrm{Confidence}
=
\mathrm{Confidence}_{\mathrm{AI\mbox{-}assist}}
-
\mathrm{Confidence}_{\mathrm{No\mbox{-}assist}}.
\]

| Group | Mean change, points | 95% interval |
|---|---:|---:|
| Rater 1 | +4.35 | +0.71 to +7.95 |
| Rater 2 | +3.53 | -0.15 to +7.07 |
| Overall | **+3.94** | **+1.52 to +6.36** |

The overall estimate indicates higher reported confidence under the assisted condition. The Rater 2 interval includes zero, so rater-specific results should not be overstated.

### 5.2 Decision-time change

Decision-time change is defined as:

\[
\Delta \mathrm{Time}
=
\mathrm{Time}_{\mathrm{AI\mbox{-}assist}}
-
\mathrm{Time}_{\mathrm{No\mbox{-}assist}}.
\]

| Group | Mean change, seconds | 95% interval |
|---|---:|---:|
| Rater 1 | -0.21 | -1.42 to +1.02 |
| Rater 2 | -0.18 | -1.52 to +1.20 |
| Overall | **-0.19** | **-1.10 to +0.79** |

The interval includes zero. The result should therefore be described as no clear evidence of a material decision-time increase or decrease in this exploratory sample.

### 5.3 Confidence–accuracy calibration

Clinician confidence–accuracy calibration was summarized with **expected calibration error using 10 equal-width bins**.

| Condition | ECE |
|---|---:|
| Unaided | 13.2% |
| AI-assisted | 7.0% |

The assisted condition showed lower ECE in this dataset. This is a descriptive within-study result and should not be generalized beyond the analyzed readers, images, and task without external validation.

### 5.4 Descriptive automation-risk audit

For AI-assisted reads, the figure reports paired correctness transitions among reads exceeding selected AI top-1 probability thresholds.

Definitions:

- **C→W:** unaided response correct, assisted response wrong;
- **W→C:** unaided response wrong, assisted response correct;
- **Net harm:**
  \[
  \frac{\#(\mathrm{C\rightarrow W})-\#(\mathrm{W\rightarrow C})}{n}\times100
  \]
  percentage points;
- **\(\Delta\) Conf:** mean assisted-minus-unaided confidence change in the thresholded subset.

| AI top-1 probability threshold | \(n\) | C→W | W→C | Net harm | Mean confidence change |
|---|---:|---:|---:|---:|---:|
| \(\geq 0.70\) | 9 | 1/9 (11.1%) | 3/9 (33.3%) | -22.2 pp | +2.4 points |
| \(\geq 0.80\) | 3 | 0/3 (0.0%) | 0/3 (0.0%) | 0.0 pp | -8.8 points |

These subsets are very small. The panel is a descriptive audit only and must not be interpreted as an independent clinical-safety analysis.

---

## 6. Row-level data file

Expected filename:

```text
readerstudy_400rows.csv
```

Expected dimensions:

```text
400 rows × 14 columns
```

### 6.1 Data dictionary

| Column | Type | Description |
|---|---|---|
| `repeat_id` | integer | Analysis repetition identifier. The released file uses one repetition. |
| `split_seed` | integer | Identifier associated with the fixed study split or generation protocol. |
| `image_id` | integer/string | De-identified image identifier. |
| `gt_grade` | integer | Reference RetinaMNIST grade. |
| `set_id` | string | Dataset partition identifier; the study records belong to the test set. |
| `clinician_id` | integer/string | De-identified rater identifier. |
| `condition` | string | `No-assist` or `AI-assist`. |
| `final_grade` | integer | Rater's final assigned grade. |
| `confidence_score` | numeric | Reported clinician confidence on the study scale. |
| `decision_time_sec` | numeric | Decision time in seconds. |
| `correct` | binary | Indicator that `final_grade == gt_grade`. |
| `ai_shown_pred_grade` | integer | AI-predicted grade associated with the image; displayed in the assisted condition. |
| `ai_shown_prob_top1` | numeric/NA | AI top-1 probability. It is populated for assisted records and may be missing in unaided rows. |
| `confidence_prob_calib` | numeric | Clinician confidence mapped to the probability scale used for calibration analysis. |

### 6.2 Required condition labels

The public analysis must use the following canonical values:

```text
No-assist
AI-assist
```

Do not silently rename one condition to `baseline`, because “baseline” is used elsewhere in the IDENGATE repository for model comparisons and can create ambiguity.

---

## 7. Data-integrity requirements

Before computing any result, the analysis must verify all of the following:

1. The file contains exactly **400 rows**.
2. There are exactly **2 raters**.
3. There are exactly **100 unique images**.
4. There are exactly **200 unique image–rater pairs**.
5. Every image–rater pair contains exactly **2 rows**.
6. Every image–rater pair contains one `No-assist` row.
7. Every image–rater pair contains one `AI-assist` row.
8. `correct` is binary.
9. `decision_time_sec` is finite and non-negative.
10. `confidence_prob_calib` lies in \([0,1]\).
11. `ai_shown_prob_top1` lies in \([0,1]\) wherever it is present.
12. Ground-truth labels are identical across the two conditions within each image–rater pair.
13. The AI prediction associated with one image is consistent across the paired records.
14. Test rows are not used to fit or tune the IDENGATE model.
15. No personally identifying rater information is present.

A minimal integrity check can follow this pattern:

```python
import pandas as pd

df = pd.read_csv("readerstudy_400rows.csv")

assert len(df) == 400
assert df["clinician_id"].nunique() == 2
assert df["image_id"].nunique() == 100
assert set(df["condition"].unique()) == {"No-assist", "AI-assist"}

pair_counts = (
    df.groupby(["image_id", "clinician_id"])
      .size()
)

assert len(pair_counts) == 200
assert pair_counts.eq(2).all()

condition_counts = (
    df.groupby(["image_id", "clinician_id", "condition"])
      .size()
      .unstack(fill_value=0)
)

assert condition_counts["No-assist"].eq(1).all()
assert condition_counts["AI-assist"].eq(1).all()
```

The production script should fail explicitly when an invariant is violated rather than continue with partially paired data.

---

## 8. Paired estimands

Construct one paired record for each `(image_id, clinician_id)` combination.

### 8.1 Confidence

\[
\Delta c_{ij}
=
c^{\mathrm{assist}}_{ij}
-
c^{\mathrm{unaided}}_{ij},
\]

where \(i\) indexes images and \(j\) indexes raters.

Report:

- mean \(\Delta c_{ij}\) for each rater;
- overall mean across the 200 paired records;
- corresponding image-clustered bootstrap intervals.

### 8.2 Decision time

\[
\Delta t_{ij}
=
t^{\mathrm{assist}}_{ij}
-
t^{\mathrm{unaided}}_{ij}.
\]

Report:

- mean \(\Delta t_{ij}\) for each rater;
- overall mean across all paired records;
- corresponding image-clustered bootstrap intervals.

### 8.3 Correctness transitions

For each paired record, classify the transition as:

- C→C;
- C→W;
- W→C;
- W→W.

Thresholded automation-risk summaries are computed after selecting assisted rows whose AI top-1 probability meets the specified threshold.

---

## 9. Bootstrap analysis

The manuscript reports **5,000 image-clustered bootstrap resamples**.

The clustering unit is `image_id`, not the individual row and not the image–rater pair. Each bootstrap draw must:

1. sample the 100 image identifiers with replacement;
2. retain both raters for every selected image;
3. retain both conditions for every selected image–rater pair;
4. preserve duplicate sampled images as duplicate clusters;
5. recompute the requested paired statistic.

This preserves the dependence induced by having both raters evaluate the same images.

Do not bootstrap the 400 rows independently. Row-level resampling would break the matched structure and produce an analysis different from the reported protocol.

The exact random seed and interval-construction rule should be recorded in the generated run manifest. The public analysis script, rather than a hand-edited figure, must be the authoritative source for the reproduced intervals.

---

## 10. Confidence–accuracy calibration

### 10.1 Input

Use:

```text
confidence_prob_calib
```

as the confidence probability for clinician calibration.

### 10.2 Binning

Use **10 equal-width bins** over \([0,1]\).

For bin \(B_m\),

\[
\operatorname{acc}(B_m)
=
\frac{1}{|B_m|}
\sum_{i\in B_m} y_i,
\]

\[
\operatorname{conf}(B_m)
=
\frac{1}{|B_m|}
\sum_{i\in B_m} p_i,
\]

where \(y_i\) is correctness and \(p_i\) is calibrated clinician confidence.

ECE is:

\[
\operatorname{ECE}
=
\sum_{m=1}^{10}
\frac{|B_m|}{n}
\left|
\operatorname{acc}(B_m)
-
\operatorname{conf}(B_m)
\right|.
\]

Compute ECE separately for `No-assist` and `AI-assist`.

The reliability diagram should display:

- empirical accuracy by bin;
- mean confidence by bin;
- the perfect-calibration diagonal;
- condition-specific ECE values.

Empty bins must not contribute to ECE.

---

## 11. Reproduction workflow

Recommended files:

```text
reader_study/
├── README.md
├── readerstudy_400rows.csv
├── analyze_reader_study.py
├── reproduce_reader_study_figure.py
├── expected_summary.json
└── outputs/
    └── .gitkeep
```

The figure remains shared at:

```text
assets/reader_study.png
```

### 11.1 Environment

Minimum analysis dependencies:

```text
numpy
pandas
scipy
matplotlib
```

Use the package versions pinned by the root IDENGATE environment where available.

### 11.2 Analysis command

```bash
python reader_study/analyze_reader_study.py \
  --input reader_study/readerstudy_400rows.csv \
  --bootstrap-resamples 5000 \
  --output-dir reader_study/outputs
```

### 11.3 Figure command

```bash
python reader_study/reproduce_reader_study_figure.py \
  --summary reader_study/outputs/reader_study_summary.json \
  --calibration reader_study/outputs/calibration_bins.csv \
  --automation-risk reader_study/outputs/automation_risk.csv \
  --output assets/reader_study.png
```

### 11.4 Expected outputs

```text
reader_study/outputs/
├── integrity_report.json
├── paired_records.csv
├── reader_study_summary.json
├── confidence_bootstrap.csv
├── time_bootstrap.csv
├── calibration_bins.csv
├── automation_risk.csv
└── run_manifest.json
```

`run_manifest.json` should record:

- source CSV SHA-256;
- source-code commit;
- Python and package versions;
- bootstrap resample count;
- bootstrap seed;
- confidence-interval rule;
- analysis timestamp;
- generated output hashes.

---

## 12. Reference verification targets

The reproduction should recover the following published values, subject only to the exact stored bootstrap seed and interval implementation.

```json
{
  "n_rows": 400,
  "n_raters": 2,
  "n_images": 100,
  "n_image_rater_pairs": 200,
  "bootstrap_resamples": 5000,
  "confidence_change": {
    "rater_1": 4.35,
    "rater_2": 3.53,
    "overall": 3.94
  },
  "decision_time_change_seconds": {
    "rater_1": -0.21,
    "rater_2": -0.18,
    "overall": -0.19
  },
  "ece": {
    "unaided": 0.132,
    "ai_assisted": 0.070
  }
}
```

The following intervals should also be reproduced:

```text
Confidence change:
Rater 1: +0.71 to +7.95
Rater 2: -0.15 to +7.07
Overall: +1.52 to +6.36

Decision-time change:
Rater 1: -1.42 to +1.02 seconds
Rater 2: -1.52 to +1.20 seconds
Overall: -1.10 to +0.79 seconds
```

Do not replace the source data or analysis outputs with manually entered target values. Expected values are validation references only.

---

## 13. Interpretation rules

Use language consistent with the exploratory design.

### Supported wording

- “AI assistance was associated with a mean confidence increase of 3.94 points.”
- “The overall bootstrap interval for confidence change was 1.52 to 6.36 points.”
- “The mean decision-time change was -0.19 seconds, with an interval spanning zero.”
- “Clinician confidence–accuracy ECE decreased from 13.2% to 7.0% in this dataset.”
- “The thresholded automation-risk analysis was descriptive and based on small subsets.”

### Avoid

- “IDENGATE is clinically safe.”
- “IDENGATE improves patient outcomes.”
- “The reader study proves diagnostic superiority.”
- “The method eliminates automation bias.”
- “The assisted condition was significantly faster.”
- “Grad-CAM evidence is causal.”
- “The study validates deployment readiness.”

The data support an exploratory paired reader analysis, not a clinical-use claim.

---

## 14. Privacy and release checklist

Before making this directory public:

- [ ] Confirm that row-level release is permitted by the IRB and consent materials.
- [ ] Confirm that `clinician_id` is de-identified.
- [ ] Remove names, emails, timestamps, IP addresses, free-text notes, and institutional identifiers.
- [ ] Confirm that public image identifiers cannot expose private records.
- [ ] Record the ethics statement exactly as **SKKU IRB No. 2026-01-011**.
- [ ] Publish only public or properly authorized image references.
- [ ] Include a SHA-256 checksum for every released data file.
- [ ] Verify that the figure can be regenerated from released or permitted aggregate data.
- [ ] Preserve the wording “exploratory retrospective reader study.”
- [ ] State that the analysis did not affect patient care.

---

## 15. Relationship to the main IDENGATE repository

The reader study is one component of the broader IDENGATE evidence package. It is separate from:

- model training;
- MGF ON versus MGF OFF fixed-checkpoint analysis;
- no-MGF and Shuffle-MGF controls;
- calibration of model probabilities;
- risk–coverage analysis;
- Grad-CAM overlap evaluation;
- perturbation robustness;
- cross-dataset performance comparisons.

Clinician confidence calibration in this directory must not be confused with the model ECE analysis reported elsewhere in the repository.

The AI-assisted condition should be described consistently with the study materials. It should not be relabeled as a separately trained model or as a clinical decision-support product.

---

## 16. Citation

Until final publication metadata are available:

```bibtex
@misc{huang2026idengate,
  title  = {IDENGATE: Bounded Role-Specific Preprojection Gating with Exact Identity Control for Medical Image Classification},
  author = {Hao Huang and Jee-Hyong Lee and Ce Gao and Ning Wang and Zongrun Sun},
  year   = {2026},
  note   = {Manuscript submitted to IEEE Transactions on Medical Imaging}
}
```

---

## 17. Disclaimer

This material is provided for research transparency and reproducibility. IDENGATE is not a certified medical device. The reader-study results must not be used to support diagnosis, patient management, regulatory claims, or clinical deployment without appropriate independent validation, prospective study design, regulatory review, and institutional approval.
