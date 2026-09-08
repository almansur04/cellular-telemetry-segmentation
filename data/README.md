# Data

The raw cellular telemetry dataset is intentionally not committed to this repository.

Expected local/Colab location:

`/content/drive/MyDrive/webbrowsing.csv`

The dataset used in the study contains 1,000,000 mobile web-browsing sessions collected by a single operator in Kazakhstan.

To reproduce the analysis in Google Colab:

1. Mount Google Drive.
2. Clone this repository.
3. Ensure `webbrowsing.csv` is located at the path defined in `configs/default.yaml`.
4. Run `python scripts/run_all.py --config configs/default.yaml`.

Do not commit proprietary, personal, subscriber-identifying, or otherwise restricted telemetry to the public repository.
