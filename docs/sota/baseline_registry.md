# Baseline Registry (versions, configs, seeds)

Implemented
- CSP+LDA
  - Script: scripts/run_csp_lda.py
  - Params: filters_per_class=3; bands 8–30 Hz (use preprocess bandpass 4–40 Hz)
  - Seeds: record via CLI env or config; per-subject runs
- EEGNet
  - Script: scripts/run_eegnet.py (torch optional)
  - Params: batch_size=64; epochs=200; lr=1e-3; patience=25 (as in RUNNERS.md)
  - Seeds: fixed seeds for reproducibility

Planned
- DeepConvNet: add scripts/run_deepconvnet.py with canonical config
- Kalman-based: add scripts/run_kalman.py (select variant, document parameters)
- EEGFormer/GNN/FM: add scripts/run_eegformer_or_gnn.py (choose a representative model)

Artifacts
- Per-subject CSV: artifacts/.../{SUBJECT}_{MODEL}.csv
- Aggregate + stats: scripts/aggregate_and_stats.py → wilcoxon_pvalues.csv
