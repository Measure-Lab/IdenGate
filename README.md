# IDENGATE README Alignment Audit

## Why the early README must be replaced

The early README is not just visually simple; several statements no longer match the submitted manuscript.

### 1. Scientific framing

The early title and abstract emphasize “evidence–confidence alignment,” “morphology-conditioned” gating, “causal structure,” and “safer human–AI decision-making.” The final manuscript instead centers on:

- bounded role-specific preprojection conditioning;
- image-specific effective Q/K/V projections;
- exact within-checkpoint identity control;
- fixed-checkpoint role interventions;
- structural non-equivalence and norm bounds;
- controlled mechanism evaluation.

The public README must not make stronger causal or clinical-safety claims than the paper.

### 2. Identity-state definition

The early README describes zero gate strength as an “unconditioned baseline.” This is incorrect.

- MGF ON: \(\alpha=0.1\), learned-gate state.
- MGF OFF: \(\alpha=0\), exact identity-gate state of the same checkpoint.
- no-MGF: separately trained model without MGF.

These three states must remain distinct in code, filenames, plots, and documentation.

### 3. Missing method correspondence

The early README omits the manuscript’s central technical definitions:

- ALME equations;
- role-specific MGF equations;
- \(Q=XD_QW_Q\), \(K=XD_KW_K\), \(V=XD_VW_V\);
- exact identity equation;
- \([0.9,1.1]\) bound at \(\alpha=0.1\);
- preprojection/postprojection non-equivalence;
- feature and score norm bounds.

### 4. Reproducibility gaps

The old README provides a single demo evaluation command, while the paper states that training, evaluation, ablation, and figure-generation code are available.

The public release therefore needs explicit entry points for:

- training;
- standard evaluation;
- no-MGF and Shuffle-MGF;
- Q/K/V identity interventions;
- perturbations and strength sweeps;
- calibration and risk–coverage;
- Grad-CAM overlap;
- Table II, Table III, and Figures 3–6.

### 5. Incorrect or low-quality repository details

The early tree contains `cmanet_blood_dp_best.pth`, which is unrelated to IDENGATE and must be removed.

The Google Drive RetinaMNIST copy should not define the canonical protocol. The paper uses official MedMNIST splits.

PyCharm recommendations, fixed installation-time claims, fixed runtime claims, and Windows-demo emphasis should be removed from the main README.

### 6. External comparator provenance

The manuscript explicitly states that external comparators were not retrained under a common implementation.

- Figure 5 uses source-reported comparator values.
- Table III uses the cited MedViTV2 benchmark values.
- Those values provide context, not protocol-matched superiority evidence.

The repository and README must label them accordingly.

## Mandatory code invariants before publication

1. `alpha=0` produces exact identity gates.
2. MGF OFF and MGF ON use the same checkpoint.
3. no-MGF is separately trained.
4. Shuffle-MGF preserves the mini-batch gate-vector multiset.
5. Q/K/V interventions change only selected gates.
6. ALME and descriptor computation remain active under MGF OFF.
7. Normalization statistics remain fixed in paired inference.
8. Perturbations occur after resize and before normalization.
9. Temperature scaling is fitted on validation logits separately for each state.
10. ECE uses 10 equal-width bins.
11. AURC uses empirical risk–coverage and trapezoidal integration.
12. Seed-level outputs are saved before aggregation.
13. RetinaMNIST uses seeds 42–46, 60 epochs, batch 128, AdamW, learning rate \(3\times10^{-4}\), weight decay \(10^{-4}\), and cosine restarts with \(T_0=10\), \(T_{\mathrm{mult}}=2\).
14. Checkpoint selection uses highest validation macro-AUC.
15. No class weighting or pretraining is used in the five-seed RetinaMNIST mechanism study.
16. Strength sweep values are \(0,0.5,1,2,4\).
17. Grad-CAM uses the 85th activation percentile.
18. External values are marked source reported or benchmark reproduced.
19. Splits and checkpoints have hashes.
20. Every paper table and figure has one reproduction entry point.

## Publication rule

The professional README is the target public-release specification. It should replace the old README only after the corresponding paths and commands exist in the refactored repository.
