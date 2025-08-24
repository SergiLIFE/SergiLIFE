# E2E Latency Testbench (External Trigger)

Goal: Measure true end-to-end latency under clinical budget (≤100ms).

Setup:
- External trigger (HTTP or WebSocket) generates timestamped requests.
- System processes through ingest→organize→learn→assess→optimize.
- Responder returns timestamp; harness computes latency and writes evidence JSON and CSV.

Artifacts:
- evidence/latency_runs.json
- evidence/latency_runs.csv
- plots/latency_trend.png

Notes:
- Run during PR and nightly to guard regressions.
- Record environment (CPU/GPU, memory) for comparability.
