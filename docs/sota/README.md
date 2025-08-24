# EEG MI SOTA Benchmark Plan (BCI IV-2a and Extensions)

This folder captures the standardized plan and pointers to code/assets for evaluating L.I.F.E against community-accepted MI baselines and protocols.

Status at a glance:
- Step 1 (Datasets):
  - BCI Competition IV 2a: Implemented (scripts + config + docs)
  - WBCIC-MI / OpenBMI: Partially prepared (aliases + placeholders) — runners pending
- Step 2 (Baselines):
  - CSP+LDA: Implemented (script + runners)
  - EEGNet: Implemented (script + runners)
  - DeepConvNet: Planned (doc stub, not yet implemented)
  - Kalman-based: Planned (doc stub, not yet implemented)
  - GNN/Transformer/Foundation model: Planned (doc stub, not yet implemented)
- Step 3 (Versions/Configs/Protocol):
  - Experiment manifest + stats: Implemented for BCI IV-2a minimal
- Step 4 (Justification):
  - Captured in this plan with references
- Step 5 (Deliverables):
  - Dataset cards: 2a added; WBCIC-MI/OpenBMI stubs suggested
  - Baseline registry: Added (mapping to current scripts)
  - Protocol sheet: Added (metrics + Wilcoxon)

Quick links
- Dataset cards: ./dataset_cards/BCI_IV_2a.md
- Baseline registry: ./baseline_registry.md
- Protocol: ./protocol.md
- Whitepaper (dated): ../whitepaper/LIFE_THEORY_Technical_Whitepaper_2025-08-23.md
- Whitepaper (main): ../whitepaper/LIFE_THEORY_Whitepaper.md
- Experiment manifest: ../../experiments/bciiv2a_sota_minimal.yaml
- Runners: ../../RUNNERS.md
- Data acquisition: ../../DATA_ACQUISITION.md

Code pointers (existing)
- Preprocess 2a: scripts/preprocess_bciiv2a.py (see RUNNERS.md)
- CSP+LDA runner: scripts/run_csp_lda.py
- EEGNet runner: scripts/run_eegnet.py
- Aggregation + Wilcoxon: scripts/aggregate_and_stats.py
- Dataset utilities and aliases: tools/bci_sota_benchmark.py, BCI_DATASETS.md

Next up (implementation TODO)
- Add: scripts/run_deepconvnet.py
- Add: scripts/run_kalman.py (feature/decoder variant per selected paper)
- Add: scripts/run_eegformer_or_gnn.py (representative model)
- Add: experiments/wbcic_mi.yaml + scripts/preprocess_wbcic_mi.py
