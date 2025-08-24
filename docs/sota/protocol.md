# Evaluation Protocol (standardized)

Datasets/Tasks
- BCI IV-2a (4-class MI): per-subject; optional cross-subject; follow bandpass/notch/CAR; 0.5–4.5 s trials.
- WBCIC-MI (multi-day): 2-class/3-class; include cross-session; mirror published baselines where applicable.

Models
- Classical: CSP+LDA (optionally SVM)
- Deep: EEGNet, DeepConvNet (planned)
- State-space: Kalman (planned)
- Frontier: EEGFormer/GNN/Foundational (planned)

Versioning & Seeds
- Record script versions (git commit), Python/lib versions, seeds per run.
- Save configs/CLI args to artifacts.

Metrics & Stats
- Per-subject: Accuracy, Macro-F1, Kappa
- Aggregate: mean±std across subjects
- Statistical test: Wilcoxon signed-rank (paired per subject) for model pairs

Reporting
- Tables: per_subject_results.csv, aggregate_summary.csv, wilcoxon_pvalues.csv
- Manifest: experiments/bciiv2a_sota_minimal.yaml
- Runners: RUNNERS.md one-liners

Reproducibility
- Data acquisition documented (DATA_ACQUISITION.md)
- Preprocessing scripts checked-in
- Fixed seeds; export CLI into artifacts
