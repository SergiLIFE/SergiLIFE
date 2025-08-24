# Dataset Card: BCI Competition IV 2a (4-class MI)

- Subjects: 9 (A01–A09)
- Channels: 22 EEG + 3 EOG
- Sampling rate: 250 Hz
- Tasks: Left hand, Right hand, Both feet, Tongue (motor imagery)
- Format: GDF
- License/Access: Competition terms; download from official site.

Preprocessing (as in this repo)
- Bandpass: ~4–40 Hz
- Notch: 50 Hz (EU)
- CAR (common average reference)
- Trial window: ~0.5–4.5 s post-cue

Repo pointers
- Data layout and acquisition: DATA_ACQUISITION.md
- Preprocess script: scripts/preprocess_bciiv2a.py
- NPZ epochs root (example): data/BCI_IV_2a_npz

Splits & Evaluation
- Per-subject training/testing; report per-subject Accuracy, Macro-F1, Kappa
- Cross-subject optional; document split policy if used

Notes
- Align settings with literature/leaderboards for comparability
- Use fixed random seeds and record versions as per baseline registry
