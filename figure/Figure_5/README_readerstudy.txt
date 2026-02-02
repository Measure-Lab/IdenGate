### Reader study CSV (anonymized) + Figure 5 reproduction

This folder provides an **anonymized** reader-study table and a script that reproduces **Figure 5 directly from the CSV** (no hidden inputs).

#### Privacy and anonymization
To protect participant privacy, the table is **de-identified**:
- No direct identifiers are included (e.g., names, emails, MRNs, exact timestamps, free-text notes).
- `reader_id` (stored as `clinician_id` in the CSV) and `image_id` are **pseudonymous IDs** used only for within-table grouping; they are not linkable outside the study team.
- Only fields needed to reproduce the reported **aggregate** results in Figure 5 are retained.
- Potentially sensitive metadata (site/hospital identifiers, original file paths, patient metadata, etc.) has been removed.

If you need additional study details (protocol, randomization, governance documents), please contact the corresponding author.

#### Files
- `readerstudy_400rows.csv`  
  Main table (one row = one reader × one image × one condition).
- `Figure5_*.py`  
  Script(s) to reproduce Figure 5 panels (a–d) **from the CSV only**.

#### Row definition
Each row corresponds to one reader’s assessment of one image under one condition:
- `condition ∈ {AI-assist, No-assist}`
- Total rows = (#readers) × (#images) × 2

#### Columns used in Figure 5
- Pairing keys: `repeat_id`, `split_seed`, `image_id`, `set_id`, `clinician_id`, `gt_grade`
- Reader outcomes: `final_grade`, `correct`, `confidence_score`, `decision_time_sec`
- AI suggestion (AI-assist only): `ai_shown_pred_grade`, `ai_shown_prob_top1`
- Calibration (optional): `confidence_prob_calib`  
  If present, panel (c) uses this (0–1). Otherwise, the script falls back to `confidence_score/100`.

#### Reproducing Figure 5
1. Install: `numpy`, `pandas`, `matplotlib`
2. Set `BASE_DIR` to this folder
3. Run: `python Figure5_...py`
4. Outputs: `Figure5_fourpanel_*.png/.pdf` (and optional per-panel images)

#### Notes
- `ai_shown_prob_top1` is expected to be empty for `No-assist` rows (AI confidence is not shown in that condition).
- Some AI-related fields may be logged for auditing even when not shown to readers.

#### Contact
For the full protocol and supplementary materials, please contact the corresponding author.
