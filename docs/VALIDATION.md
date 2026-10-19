# Release checks — 2026-10-19

The final release was checked on Windows CPU using Python 3.11.9, PyTorch 2.10.0+cpu, NumPy 2.2.6 and RDKit 2024.09.6. The pinned benchmark training environment remains the Python 3.10 stack in the requirements files; these software checks do not constitute a new benchmark run.

- 21 unit tests passed, including observed-label masking, exact AP/AUROC threshold grouping with ties, feature invariance, training statistics, a five-epoch synthetic training smoke test, all pathway switches, raw-probability output, explicit evaluation alignment and fixed similarity boundaries.
- All nine full/pathway/interaction configurations produced exactly the same CPU logits as the manuscript reference implementation on matched synthetic inputs and weights.
- All ten archived study heads loaded with unchanged state keys and matched the reference forward outputs exactly on the same CPU inputs. Those weights remain outside this public repository.
- The real score snapshot passed mean/sample-SD, paired-ablation, external-cohort and shared-domain-eligibility checks: 110 main partition scores, 45 external scores and 900 similarity-stratified records. SHA256 values are recorded in `results/manifest.json`.
- The Windows and Linux feature-definition files differ only in line endings. Verification normalizes CRLF to LF before comparing the locked definition hash, and feature parsing uses the verified text so Unicode installation paths work. The generated feature families and atom memberships were compared against RDKit's file-based factory.

No graph model or benchmark was retrained for this code synchronization. Existing data partitions, selected experimental predictions and model hyperparameters remain unchanged. Test inputs are synthetic and are never included in the manuscript score files.
