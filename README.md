# Cellular Telemetry Segmentation

A reproducible analysis pipeline for studying device-model heterogeneity and cell-level quality impact in large-scale mobile web telemetry.

## Research objective

The project asks two related questions:

1. Do device models exhibit different application-level throughput distributions after restricting workload and operating conditions?
2. Can cell-level quality failures be ranked by observed user-impact volume to support network investigation?

The methodology intentionally distinguishes:

- measured device-model differences;
- application/workload heterogeneity;
- failure severity;
- failure impact;
- operational triage candidates.

It does not claim causal root-cause diagnosis from telemetry alone.

## Dataset

The study uses one million mobile web-browsing sessions collected by a single cellular operator in Kazakhstan.

The raw dataset is not included in this repository.

Expected Colab path:

`/content/drive/MyDrive/webbrowsing.csv`

## Reproduction in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
