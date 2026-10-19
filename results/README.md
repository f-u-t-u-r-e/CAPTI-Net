# Final main-manuscript score snapshot — 2026-10-19

All predictions are standalone. Random forest and AttentiveFP are excluded from the current main comparison at the authors' request. No remaining fold, assay or chemical-similarity bin is selected by its outcome. Scores were copied from the verified main source data without rerunning or selecting models.

| File | Contents |
|---|---|
| `partition_scores.csv` | Eleven retained methods × two panels × five scaffold partitions; AP/AUROC in [0,1] |
| `table1.csv` | Unrounded means, sample SD and reported parameter counts |
| `assay_means.csv` | All eligible assay means, sample SD and partition coverage for the neural methods |
| `validation_controls.csv` | Original neural validation controls; the main checkmark table uses the full model and four pathway/product removals |
| `pathway_removal.csv` | Matched full-minus-removal test results for three pathways |
| `single_pathways.csv` | Matched full-minus-single-pathway test results |
| `external_herg.csv` | Three predictors × external I/II/III × five source models |
| `similarity_per_seed.csv` | Neural peers and pathway controls; bin-specific and common-assay results |

Table 1 includes the final 5-layer, width-300 supervised GINE author-configuration rerun. FewTox parameter counts cover trainable prediction heads only; other neural counts include all weights used for prediction, including required frozen encoders. These different accounting scopes are explicit in the manuscript.

AP is average precision, not trapezoidal PR-AUC. Macro metrics weight eligible assays equally. Across partitions, SD uses `ddof=1`; partitions overlap. External SD describes five source fits evaluated on the same cohort. No confidence intervals or significance claims are added.

`similarity_per_seed.csv` uses proportions for peer metrics and **percentage points** for `delta_*` pathway rows. Its historical `without_*` identifiers name omitted pathways. `macro_ap_skill`, where retained, is an auxiliary prevalence-normalized metric and is not used in the main figures.

The strongest-main-comparator AP differences are 3.270 pp on Tox21 and 1.266 pp on ToxCast; AUROC differences are 0.935 and 1.374 pp. CAPTI-Net leads AP/AUROC on hERG I/III and AUROC on II. On II, D-MPNN has the higher mean AP. In the domain analysis, CAPTI-Net leads both metrics over D-MPNN and MolCLR in all four fixed bins on both panels, under both eligibility definitions.

The model package contains no comparator training code or pretrained weights. These archived scores document the experiments; training the released method with another runtime is a new run and is not expected to reproduce every decimal bit-for-bit. Run `python scripts/verify_results.py` to verify the saved score relationships.
