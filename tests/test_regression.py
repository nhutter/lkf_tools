"""
Regression test: re-runs detection and tracking on the McGill sample file
and asserts that output matches the frozen reference fixture.

Run with:
    conda run -n lkf_tools pytest tests/test_regression.py
"""

from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NETCDF_FILE = REPO_ROOT / "data" / "McGill" / "McGill_e2_1997_daily_means.nc"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "reference_output.npz"


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Run detection + tracking once for the whole module and return results."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lkf_tools.dataset import process_dataset

    out_dir = tmp_path_factory.mktemp("lkfs")
    lkf_data = process_dataset(str(NETCDF_FILE), output_path=str(out_dir) + "/")
    lkf_data.detect_lkfs(indexes=[0, 1])
    lkf_data.track_lkfs(indexes=[0])

    stem = NETCDF_FILE.stem
    lkfs_0 = np.load(lkf_data.lkfpath / f"lkf_{stem}_001.npy", allow_pickle=True)
    lkfs_1 = np.load(lkf_data.lkfpath / f"lkf_{stem}_002.npy", allow_pickle=True)

    track_files = list(lkf_data.track_output_path.glob("lkf_tracked_pairs_*.npy"))
    assert len(track_files) == 1
    tracked_pairs = np.load(track_files[0], allow_pickle=True)

    timestamps = np.array(lkf_data.time.values[[0, 1]], dtype="datetime64[ns]")

    return dict(
        lkfs_0=lkfs_0,
        lkfs_1=lkfs_1,
        tracked_pairs=tracked_pairs,
        timestamps=timestamps,
    )


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURE, allow_pickle=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_lkf_arrays_equal(actual, expected, label):
    assert len(actual) == len(expected), (
        f"{label}: segment count mismatch — got {len(actual)}, expected {len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.shape == e.shape, (
            f"{label}[{i}]: shape mismatch — got {a.shape}, expected {e.shape}"
        )
        # Columns 0–1 are pixel row/col indices — must be exact
        np.testing.assert_array_equal(
            a[:, :2].astype(int), e[:, :2].astype(int),
            err_msg=f"{label}[{i}]: pixel index mismatch",
        )
        # Columns 2–6 are lon, lat, div, shr, vor — allow float tolerance
        np.testing.assert_allclose(
            a[:, 2:], e[:, 2:], rtol=1e-5, atol=1e-10,
            err_msg=f"{label}[{i}]: float value mismatch",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_timestamps(pipeline_output, reference):
    np.testing.assert_array_equal(
        pipeline_output["timestamps"], reference["timestamps"],
        err_msg="timestamps mismatch",
    )


def test_lkf_count_day0(pipeline_output, reference):
    assert len(pipeline_output["lkfs_0"]) == len(reference["lkfs_0"]), (
        f"Day-0 segment count: got {len(pipeline_output['lkfs_0'])}, "
        f"expected {len(reference['lkfs_0'])}"
    )


def test_lkf_count_day1(pipeline_output, reference):
    assert len(pipeline_output["lkfs_1"]) == len(reference["lkfs_1"]), (
        f"Day-1 segment count: got {len(pipeline_output['lkfs_1'])}, "
        f"expected {len(reference['lkfs_1'])}"
    )


def test_lkf_segments_day0(pipeline_output, reference):
    assert_lkf_arrays_equal(pipeline_output["lkfs_0"], reference["lkfs_0"], "lkfs_0")


def test_lkf_segments_day1(pipeline_output, reference):
    assert_lkf_arrays_equal(pipeline_output["lkfs_1"], reference["lkfs_1"], "lkfs_1")


def test_tracked_pairs(pipeline_output, reference):
    np.testing.assert_array_equal(
        pipeline_output["tracked_pairs"], reference["tracked_pairs"],
        err_msg="tracked_pairs mismatch",
    )
