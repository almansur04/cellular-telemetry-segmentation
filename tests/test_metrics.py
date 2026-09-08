import numpy as np
import pandas as pd

from src.cell_analysis import aggregate_cells
from src.device_analysis import cliffs_delta


def test_cliffs_delta_identical_groups():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])

    assert abs(cliffs_delta(x, y)) < 1e-12


def test_aggregate_cells_threshold():
    df = pd.DataFrame(
        {
            "cell_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "id": range(8),
            "signal_strength": [-80, -81, -82, -83, -100, -101, -102, -103],
            "sla_failure": [1, 1, 0, 0, 1, 1, 1, 0],
        }
    )

    result = aggregate_cells(
        df,
        min_sessions=4,
        failure_rate_threshold=0.05,
        coverage_rsrp_threshold_dbm=-95,
    )

    assert len(result) == 2
    assert result["sla_failures"].sum() == 5
