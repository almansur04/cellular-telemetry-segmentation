import numpy as np
import pandas as pd

from src.cell_analysis import (
    aggregate_cells,
    jaccard_similarity,
    wilson_interval,
)
from src.device_analysis import cliffs_delta


def test_cliffs_delta_identical_groups():
    x = np.array(
        [1, 2, 3]
    )

    y = np.array(
        [1, 2, 3]
    )

    assert (
        abs(
            cliffs_delta(
                x,
                y,
            )
        )
        < 1e-12
    )


def test_wilson_interval_is_bounded():
    low, high = wilson_interval(
        failures=5,
        total=100,
    )

    assert (
        0
        <= low
        <= high
        <= 1
    )


def test_aggregate_cells_threshold():
    df = pd.DataFrame(
        {
            "cell_id": [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
            ],
            "id": range(8),
            "signal_strength": [
                -80,
                -81,
                -82,
                -83,
                -100,
                -101,
                -102,
                -103,
            ],
            "quality_failure": [
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
            ],
        }
    )

    result = aggregate_cells(
        df,
        min_sessions=4,
        failure_rate_threshold=0.05,
        coverage_rsrp_threshold_dbm=-95,
    )

    assert len(result) == 2

    assert (
        result[
            "quality_failures"
        ].sum()
        == 5
    )


def test_jaccard_similarity():
    a = {
        1,
        2,
        3,
    }

    b = {
        2,
        3,
        4,
    }

    assert np.isclose(
        jaccard_similarity(
            a,
            b,
        ),
        2 / 4,
    )

def test_wilson_interval_zero_failures():
    from src.cell_analysis import wilson_interval

    low, high = wilson_interval(
        failures=0,
        total=100,
    )

    assert np.isclose(
        low,
        0.0,
        atol=1e-12,
    )

    assert 0.0 < high < 0.10


def test_cell_classification_changes_with_rsrp_threshold():
    from src.cell_analysis import (
        add_session_failures,
        aggregate_cells,
    )

    df = pd.DataFrame(
        {
            "cell_id": [1, 1, 2, 2],
            "id": [1, 2, 3, 4],
            "signal_strength": [
                -90,
                -90,
                -100,
                -100,
            ],
            "page_response_latency": [
                2000,
                2000,
                2000,
                2000,
            ],
            "page_download_speed": [
                2.0,
                2.0,
                2.0,
                2.0,
            ],
            "speed_test_done": [
                1,
                1,
                1,
                1,
            ],
            "url": [
                "https://yandex.kz/",
                "https://yandex.kz/",
                "https://yandex.kz/",
                "https://yandex.kz/",
            ],
        }
    )

    df = add_session_failures(
        df,
        latency_failure_ms=1500,
        throughput_failure_mbps=0.5,
        throughput_urls=[
            "https://yandex.kz/",
        ],
    )

    result = aggregate_cells(
        df,
        min_sessions=2,
        failure_rate_threshold=0.05,
        coverage_rsrp_threshold_dbm=-95,
    )

    classes = dict(
        zip(
            result["cell_id"],
            result["triage_class"],
        )
    )

    assert (
        classes[1]
        == "Capacity/congestion candidate"
    )

    assert (
        classes[2]
        == "Coverage candidate"
    )