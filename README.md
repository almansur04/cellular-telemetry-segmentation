# Cellular Telemetry Segmentation

Reproducible analysis of large-scale mobile web telemetry for workload normalization, device-model comparison, cell-level quality analysis, and operational prioritization.

## Research objective

The project investigates two related questions:

1. Do device models exhibit different application-level throughput distributions after restricting workload and operating conditions?
2. Can cell-level quality failures be ranked by observed impact volume to support network investigation?

The analysis distinguishes:

- workload heterogeneity;
- device-model associations;
- failure severity;
- observed failure impact;
- operational triage candidates.

The study does not claim causal hardware attribution or automatic network root-cause diagnosis from telemetry alone.

## Dataset

The analysis uses 1,000,000 mobile web-browsing sessions collected by a single cellular operator in Kazakhstan.

The raw dataset is intentionally excluded from GitHub.

Expected local path:

`data/webbrowsing.csv`

## Reproduction

From the repository root:

```powershell
# Install the pinned minimum dependency set
pip install -r requirements.txt

# Run the complete analysis pipeline
python scripts/run_all.py --config configs/default.yaml

# Compare device-model throughput distributions
python scripts/run_device_analysis.py --config configs/default.yaml

# Run workload- and condition-adjusted device analysis
python scripts/run_adjusted_device_analysis.py --config configs/default.yaml

# Execute machine-learning analysis
python scripts/run_ml_analysis.py --config configs/default.yaml

# Analyze cell-level quality and operational impact
python scripts/run_cell_analysis.py --config configs/default.yaml

# Evaluate robustness across configured sensitivity conditions
python scripts/run_sensitivity_analysis.py --config configs/default.yaml
