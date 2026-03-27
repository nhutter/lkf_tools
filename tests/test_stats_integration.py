"""
Tests for statistical integration: coordinate projection and stat methods on lkf_dataset.

Run with:
    conda run -n lkf_tools pytest tests/test_stats_integration.py
"""

from pathlib import Path
import numpy as np
import pytest
from pyproj import Proj

REPO_ROOT = Path(__file__).resolve().parents[1]
NETCDF_FILE = REPO_ROOT / "data" / "McGill" / "McGill_e2_1997_daily_means.nc"

_DEFAULT_PROJ = Proj(proj='stere', lat_0=90, lat_ts=75, lon_0=-45, ellps='WGS84')


@pytest.fixture(scope="module")
def detected_dataset(tmp_path_factory):
    """Run detection on two timesteps + tracking, and return the lkf_dataset instance."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lkf_tools.dataset import process_dataset

    out_dir = tmp_path_factory.mktemp("lkfs_stats")
    lkf_data = process_dataset(str(NETCDF_FILE), output_path=str(out_dir) + "/")
    lkf_data.detect_lkfs(indexes=[0, 1])
    lkf_data.track_lkfs(indexes=[0])
    return lkf_data


# ---------------------------------------------------------------------------
# Phase A: coordinate projection
# ---------------------------------------------------------------------------

def test_projected_coords_column_count(detected_dataset):
    """After compute_projected_coords(), every segment must have 9 rows."""
    ds = detected_dataset
    ds.compute_projected_coords()

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    assert len(detected_ts) > 0, "No detected timesteps found"

    for t in detected_ts:
        for i, seg in enumerate(ds.lkfs[t]):
            assert seg.shape[0] == 9, (
                f"Segment {i} at timestep {t}: expected 9 rows, got {seg.shape[0]}"
            )


def test_projected_coords_index_pointers(detected_dataset):
    """compute_projected_coords() must set indm0=7, indm1=8 on the dataset."""
    ds = detected_dataset
    ds.compute_projected_coords()
    assert ds.indm0 == 7
    assert ds.indm1 == 8


def test_projected_coords_values(detected_dataset):
    """x,y appended by compute_projected_coords() must match a manual pyproj call."""
    ds = detected_dataset
    ds.compute_projected_coords()

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    # Check the first segment of the first detected timestep
    t = detected_ts[0]
    seg = ds.lkfs[t][0]  # shape (9, N_points)

    lon = seg[2, :]
    lat = seg[3, :]
    x_expected, y_expected = _DEFAULT_PROJ(lon, lat)

    np.testing.assert_allclose(seg[7, :], x_expected, rtol=1e-6,
                               err_msg="x-coordinate mismatch")
    np.testing.assert_allclose(seg[8, :], y_expected, rtol=1e-6,
                               err_msg="y-coordinate mismatch")


def test_projected_coords_idempotent(detected_dataset):
    """Calling compute_projected_coords() twice must not add extra columns."""
    ds = detected_dataset
    ds.compute_projected_coords()
    ds.compute_projected_coords()  # second call

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        for seg in ds.lkfs[t]:
            assert seg.shape[0] == 9, (
                f"Expected 9 rows after double call, got {seg.shape[0]}"
            )


def test_projected_coords_custom_proj(detected_dataset):
    """A custom Proj object passed to compute_projected_coords() is stored and used."""
    ds = detected_dataset
    custom_proj = Proj(proj='stere', lat_0=90, lat_ts=70, lon_0=0, ellps='WGS84')
    ds.compute_projected_coords(proj=custom_proj)

    assert ds.proj is custom_proj

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    t = detected_ts[0]
    seg = ds.lkfs[t][0]
    lon, lat = seg[2, :], seg[3, :]
    x_expected, y_expected = custom_proj(lon, lat)

    np.testing.assert_allclose(seg[7, :], x_expected, rtol=1e-6)
    np.testing.assert_allclose(seg[8, :], y_expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Phase B batch 1: lkf_lengths
# ---------------------------------------------------------------------------

def test_lkf_lengths_shape(detected_dataset):
    """lkf_lengths produces one length array per timestep slot."""
    from lkf_tools.stats import lkf_lengths
    ds = detected_dataset
    result = lkf_lengths(ds)

    assert len(result.lkf_length) == len(ds.lkfs)


def test_lkf_lengths_count(detected_dataset):
    """Each length array has one entry per segment in that timestep."""
    from lkf_tools.stats import lkf_lengths
    ds = detected_dataset
    result = lkf_lengths(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        assert len(result.lkf_length[t]) == len(ds.lkfs[t]), (
            f"timestep {t}: expected {len(ds.lkfs[t])} lengths, "
            f"got {len(result.lkf_length[t])}"
        )


def test_lkf_lengths_positive(detected_dataset):
    """All segment lengths must be strictly positive."""
    from lkf_tools.stats import lkf_lengths
    ds = detected_dataset
    result = lkf_lengths(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        assert np.all(result.lkf_length[t] > 0), f"Non-positive length at timestep {t}"


def test_lkf_lengths_values(detected_dataset):
    """Computed lengths must match manual arc-length calculation."""
    from lkf_tools.stats import lkf_lengths
    ds = detected_dataset
    result = lkf_lengths(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    t = detected_ts[0]
    for i, seg in enumerate(ds.lkfs[t]):
        expected = np.sum(np.sqrt(np.diff(seg[ds.indm0, :])**2 +
                                  np.diff(seg[ds.indm1, :])**2))
        np.testing.assert_allclose(result.lkf_length[t][i], expected, rtol=1e-10,
                                   err_msg=f"Length mismatch for segment {i} at timestep {t}")


def test_compute_lengths_method(detected_dataset):
    """ds.compute_lengths() caches result on ds.lengths."""
    ds = detected_dataset
    lengths = ds.compute_lengths()

    assert hasattr(ds, 'lengths'), "compute_lengths() must cache result on ds.lengths"
    assert lengths is ds.lengths


# ---------------------------------------------------------------------------
# Phase B batch 1: lkf_density
# ---------------------------------------------------------------------------

def test_lkf_density_shape(detected_dataset):
    """lkf_density produces an array with shape (n_timesteps, nx, ny)."""
    from lkf_tools.stats import lkf_density
    ds = detected_dataset
    result = lkf_density(ds, res=50e3)

    assert result.lkf_density.shape[0] == len(ds.lkfs)
    assert result.lkf_density.shape[1] == result.xedg.size - 1
    assert result.lkf_density.shape[2] == result.yedg.size - 1


def test_lkf_density_point_count(detected_dataset):
    """Sum of density grid equals total number of projected segment points."""
    from lkf_tools.stats import lkf_density
    ds = detected_dataset
    result = lkf_density(ds, res=50e3)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        total_pts = sum(seg.shape[1] for seg in ds.lkfs[t])
        grid_sum = int(result.lkf_density[t].sum())
        assert grid_sum == total_pts, (
            f"timestep {t}: density sum {grid_sum} != point count {total_pts}"
        )


def test_lkf_density_nonnegative(detected_dataset):
    """All density values must be non-negative."""
    from lkf_tools.stats import lkf_density
    ds = detected_dataset
    result = lkf_density(ds, res=50e3)

    assert np.all(result.lkf_density >= 0)


def test_compute_density_method(detected_dataset):
    """ds.compute_density() caches result on ds.density."""
    ds = detected_dataset
    density = ds.compute_density(grid_res=50e3)

    assert hasattr(ds, 'density'), "compute_density() must cache result on ds.density"
    assert density is ds.density


# ---------------------------------------------------------------------------
# Phase B batch 2: lkf_curvature
# ---------------------------------------------------------------------------

def test_lkf_curvature_shape(detected_dataset):
    """lkf_curvature produces one value per segment per timestep."""
    from lkf_tools.stats import lkf_curvature
    ds = detected_dataset
    result = lkf_curvature(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        assert len(result.lkf_curvature[t]) == len(ds.lkfs[t]), (
            f"timestep {t}: expected {len(ds.lkfs[t])} values, "
            f"got {len(result.lkf_curvature[t])}"
        )


def test_lkf_curvature_nonnegative(detected_dataset):
    """All end-to-end distances must be non-negative."""
    from lkf_tools.stats import lkf_curvature
    ds = detected_dataset
    result = lkf_curvature(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        assert np.all(np.array(result.lkf_curvature[t]) >= 0), (
            f"Negative end-to-end distance at timestep {t}"
        )


def test_lkf_curvature_leq_length(detected_dataset):
    """End-to-end distance must be <= arc length for every segment."""
    from lkf_tools.stats import lkf_curvature, lkf_lengths
    ds = detected_dataset
    curv = lkf_curvature(ds)
    lens = lkf_lengths(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        c = np.array(curv.lkf_curvature[t])
        l = lens.lkf_length[t]
        assert np.all(c <= l + 1e-6), (
            f"timestep {t}: end-to-end distance exceeds arc length"
        )


def test_lkf_curvature_values(detected_dataset):
    """Curvature values must match manual endpoint-distance calculation."""
    from lkf_tools.stats import lkf_curvature
    ds = detected_dataset
    result = lkf_curvature(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    t = detected_ts[0]
    for i, seg in enumerate(ds.lkfs[t]):
        expected = np.sqrt((seg[ds.indm0, 0] - seg[ds.indm0, -1])**2 +
                           (seg[ds.indm1, 0] - seg[ds.indm1, -1])**2)
        np.testing.assert_allclose(result.lkf_curvature[t][i], expected, rtol=1e-10,
                                   err_msg=f"Curvature mismatch for segment {i} at timestep {t}")


def test_compute_curvature_method(detected_dataset):
    """ds.compute_curvature() caches result on ds.curvature."""
    ds = detected_dataset
    curvature = ds.compute_curvature()

    assert hasattr(ds, 'curvature'), "compute_curvature() must cache result on ds.curvature"
    assert curvature is ds.curvature


# ---------------------------------------------------------------------------
# Phase B batch 2: lkf_deformation
# ---------------------------------------------------------------------------

def test_lkf_deformation_shape(detected_dataset):
    """lkf_deformation produces a [div, shr] pair per segment per timestep."""
    from lkf_tools.stats import lkf_deformation
    ds = detected_dataset
    result = lkf_deformation(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        assert len(result.lkf_deformation[t]) == len(ds.lkfs[t]), (
            f"timestep {t}: segment count mismatch"
        )
        for i, pair in enumerate(result.lkf_deformation[t]):
            assert len(pair) == 2, f"timestep {t}, segment {i}: expected [div, shr] pair"


def test_lkf_deformation_values(detected_dataset):
    """Deformation values must match manual mean(div) and mean(shr)."""
    from lkf_tools.stats import lkf_deformation
    ds = detected_dataset
    result = lkf_deformation(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    t = detected_ts[0]
    for i, seg in enumerate(ds.lkfs[t]):
        expected_div = np.mean(seg[ds.indd0, :])
        expected_shr = np.mean(seg[ds.indd1, :])
        np.testing.assert_allclose(result.lkf_deformation[t][i][0], expected_div, rtol=1e-10,
                                   err_msg=f"div mismatch segment {i} timestep {t}")
        np.testing.assert_allclose(result.lkf_deformation[t][i][1], expected_shr, rtol=1e-10,
                                   err_msg=f"shr mismatch segment {i} timestep {t}")


def test_compute_deformation_method(detected_dataset):
    """ds.compute_deformation() caches result on ds.deformation."""
    ds = detected_dataset
    deformation = ds.compute_deformation()

    assert hasattr(ds, 'deformation'), "compute_deformation() must cache result on ds.deformation"
    assert deformation is ds.deformation


# ---------------------------------------------------------------------------
# Phase B batch 3: lkf_intersection
# ---------------------------------------------------------------------------

def test_lkf_intersection_angles_range(detected_dataset):
    """All intersection angles must lie in [0, 180] degrees.

    With use_vorticity=True (default), oriented angles span [0, 180].
    """
    from lkf_tools.stats import lkf_intersection
    ds = detected_dataset
    result = lkf_intersection(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        angles = result.lkf_interc[t]
        if len(angles) == 0:
            continue
        arr = np.array(angles)
        assert np.all(arr >= 0) and np.all(arr <= 180), (
            f"timestep {t}: angles out of [0, 180]: min={arr.min():.2f} max={arr.max():.2f}"
        )


def test_lkf_intersection_pairs_valid(detected_dataset):
    """Intersection pair indices must be valid segment indices."""
    from lkf_tools.stats import lkf_intersection
    ds = detected_dataset
    result = lkf_intersection(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        n_segs = len(ds.lkfs[t])
        for pair in result.lkf_interc_par[t]:
            assert 0 <= pair[0] < n_segs, f"timestep {t}: invalid pair index {pair[0]}"
            assert 0 <= pair[1] < n_segs, f"timestep {t}: invalid pair index {pair[1]}"
            assert pair[0] < pair[1], f"timestep {t}: pairs must be ordered"


def test_lkf_intersection_angle_pair_count(detected_dataset):
    """Number of angles must equal number of pairs."""
    from lkf_tools.stats import lkf_intersection
    ds = detected_dataset
    result = lkf_intersection(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        assert len(result.lkf_interc[t]) == len(result.lkf_interc_par[t]), (
            f"timestep {t}: {len(result.lkf_interc[t])} angles but "
            f"{len(result.lkf_interc_par[t])} pairs"
        )


def test_compute_intersection_method(detected_dataset):
    """ds.compute_intersection() caches result on ds.intersection."""
    ds = detected_dataset
    inter = ds.compute_intersection()

    assert hasattr(ds, 'intersection'), "compute_intersection() must cache on ds.intersection"
    assert inter is ds.intersection


# ---------------------------------------------------------------------------
# Phase B batch 3: lkf_orientation
# ---------------------------------------------------------------------------

def test_lkf_orientation_shape(detected_dataset):
    """lkf_orientation produces one angle per segment per timestep."""
    from lkf_tools.stats import lkf_orientation
    ds = detected_dataset
    result = lkf_orientation(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        assert len(result.lkf_orientation[t]) == len(ds.lkfs[t]), (
            f"timestep {t}: expected {len(ds.lkfs[t])} angles, "
            f"got {len(result.lkf_orientation[t])}"
        )


def test_lkf_orientation_range(detected_dataset):
    """All orientation angles must lie in (-90, 90] degrees."""
    from lkf_tools.stats import lkf_orientation
    ds = detected_dataset
    result = lkf_orientation(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    for t in detected_ts:
        arr = result.lkf_orientation[t]
        assert np.all(arr > -90) and np.all(arr <= 90), (
            f"timestep {t}: angles out of (-90, 90]: min={arr.min():.2f} max={arr.max():.2f}"
        )


def test_lkf_orientation_values(detected_dataset):
    """Orientation angles must match manual lkf_poly_fit_p calculation."""
    from lkf_tools.stats import lkf_orientation, lkf_poly_fit_p
    ds = detected_dataset
    result = lkf_orientation(ds)

    detected_ts = [t for t in range(len(ds.lkfs)) if ds.lkfs[t] is not None]
    t = detected_ts[0]
    for i, seg in enumerate(ds.lkfs[t]):
        p_x, p_y = lkf_poly_fit_p(seg[ds.indm0, :], seg[ds.indm1, :], 1)
        p = p_y[0] / p_x[0]
        expected = 90. if np.isnan(p) else np.arctan(p) / np.pi * 180.
        np.testing.assert_allclose(result.lkf_orientation[t][i], expected, rtol=1e-10,
                                   err_msg=f"Orientation mismatch segment {i} timestep {t}")


def test_compute_orientation_method(detected_dataset):
    """ds.compute_orientation() caches result on ds.orientation."""
    ds = detected_dataset
    orient = ds.compute_orientation()

    assert hasattr(ds, 'orientation'), "compute_orientation() must cache on ds.orientation"
    assert orient is ds.orientation


# ---------------------------------------------------------------------------
# Phase B batch 4: lkf_lifetime + lkf_growthrate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tracked_dataset(tmp_path_factory):
    """Detect 3 timesteps and run tracking on 2 pairs; return the dataset."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lkf_tools.dataset import process_dataset

    out_dir = tmp_path_factory.mktemp("lkfs_tracked")
    ds = process_dataset(str(NETCDF_FILE), output_path=str(out_dir) + "/")
    ds.detect_lkfs(indexes=[0, 1, 2])
    ds.track_lkfs(indexes=[0, 1])
    return ds


def test_lkf_lifetime_shape(tracked_dataset):
    """lkf_lifetime has one entry per timestep; each entry length == n_segments."""
    from lkf_tools.stats import lkf_lifetime
    ds = tracked_dataset
    result = lkf_lifetime(ds)

    for t in range(len(ds.lkfs)):
        if ds.lkfs[t] is None:
            continue
        assert len(result.lkf_lifetime[t]) == len(ds.lkfs[t]), (
            f"timestep {t}: expected {len(ds.lkfs[t])} lifetimes, "
            f"got {len(result.lkf_lifetime[t])}"
        )


def test_lkf_lifetime_min(tracked_dataset):
    """Every lifetime must be >= 1."""
    from lkf_tools.stats import lkf_lifetime
    ds = tracked_dataset
    result = lkf_lifetime(ds)

    for t in range(len(ds.lkfs)):
        if ds.lkfs[t] is None:
            continue
        assert np.all(result.lkf_lifetime[t] >= 1), (
            f"timestep {t}: lifetime < 1 found"
        )


def test_lkf_lifetime_tracked_segments(tracked_dataset):
    """After tracking, at least one segment should have lifetime > 1."""
    from lkf_tools.stats import lkf_lifetime
    ds = tracked_dataset
    result = lkf_lifetime(ds)

    # Timestep 1 and 2 may have tracked segments
    any_tracked = False
    for t in range(1, len(ds.lkfs)):
        if ds.lkfs[t] is not None and np.any(result.lkf_lifetime[t] > 1):
            any_tracked = True
            break
    assert any_tracked, "Expected at least one segment with lifetime > 1 after tracking"


def test_compute_lifetime_method(tracked_dataset):
    """ds.compute_lifetime() returns and caches result on ds.lifetime."""
    ds = tracked_dataset
    result = ds.compute_lifetime()

    assert hasattr(ds, 'lifetime'), "compute_lifetime() must cache on ds.lifetime"
    assert result is ds.lifetime


def test_lkf_growthrate_shape(tracked_dataset):
    """lkf_growth and lkf_shrink have one entry per timestep; each length == n_segments."""
    from lkf_tools.stats import lkf_growthrate
    ds = tracked_dataset
    if not hasattr(ds, 'lifetime'):
        ds.compute_lifetime()
    if not hasattr(ds, 'lengths'):
        ds.compute_lengths()
    result = lkf_growthrate(ds)

    for t in range(len(ds.lkfs)):
        if ds.lkfs[t] is None:
            continue
        assert len(result.lkf_growth[t]) == len(ds.lkfs[t]), (
            f"timestep {t}: growth length {len(result.lkf_growth[t])} != n_segs {len(ds.lkfs[t])}"
        )
        assert len(result.lkf_shrink[t]) == len(ds.lkfs[t]), (
            f"timestep {t}: shrink length {len(result.lkf_shrink[t])} != n_segs {len(ds.lkfs[t])}"
        )


def test_lkf_growthrate_nonneg(tracked_dataset):
    """Non-NaN growth/shrink values must be >= 0."""
    from lkf_tools.stats import lkf_growthrate
    ds = tracked_dataset
    result = lkf_growthrate(ds)

    for t in range(len(ds.lkfs)):
        if ds.lkfs[t] is None:
            continue
        g = result.lkf_growth[t]
        s = result.lkf_shrink[t]
        assert np.all(g[~np.isnan(g)] >= 0), f"timestep {t}: negative growth found"
        assert np.all(s[~np.isnan(s)] >= 0), f"timestep {t}: negative shrink found"


def test_compute_growthrate_method(tracked_dataset):
    """ds.compute_growthrate() returns and caches result on ds.growthrate."""
    ds = tracked_dataset
    result = ds.compute_growthrate()

    assert hasattr(ds, 'growthrate'), "compute_growthrate() must cache on ds.growthrate"
    assert result is ds.growthrate


# ---------------------------------------------------------------------------
# Phase D: lkf_collection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_datasets(tmp_path_factory):
    """Two lkf_dataset instances each detecting one timestep (different indices)."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lkf_tools.dataset import process_dataset

    out_a = tmp_path_factory.mktemp("lkfs_col_a")
    ds_a = process_dataset(str(NETCDF_FILE), output_path=str(out_a) + "/")
    ds_a.detect_lkfs(indexes=[0])

    out_b = tmp_path_factory.mktemp("lkfs_col_b")
    ds_b = process_dataset(str(NETCDF_FILE), output_path=str(out_b) + "/")
    ds_b.detect_lkfs(indexes=[1])

    return ds_a, ds_b


def test_collection_construct(two_datasets):
    """lkf_collection accepts a list of lkf_dataset instances."""
    from lkf_tools.dataset import lkf_collection
    ds_a, ds_b = two_datasets
    col = lkf_collection([ds_a, ds_b])
    assert len(col.datasets) == 2


def test_collection_compute_lengths(two_datasets):
    """compute_lengths() on collection caches result on each dataset."""
    from lkf_tools.dataset import lkf_collection
    ds_a, ds_b = two_datasets
    col = lkf_collection([ds_a, ds_b])
    col.compute_lengths()
    assert hasattr(ds_a, 'lengths'), "lengths not cached on ds_a"
    assert hasattr(ds_b, 'lengths'), "lengths not cached on ds_b"


def test_collection_lkf_length_flat(two_datasets):
    """col.lkf_length matches manual concatenation from both datasets."""
    from lkf_tools.dataset import lkf_collection
    ds_a, ds_b = two_datasets
    col = lkf_collection([ds_a, ds_b])
    col.compute_lengths()

    manual = []
    for ds in [ds_a, ds_b]:
        for t in range(len(ds.lkfs)):
            if ds.lkfs[t] is not None:
                manual.append(ds.lengths.lkf_length[t])
    manual = np.concatenate(manual)

    np.testing.assert_array_equal(col.lkf_length, manual)


def test_collection_compute_density(two_datasets):
    """compute_density() on collection caches density on each dataset."""
    from lkf_tools.dataset import lkf_collection
    ds_a, ds_b = two_datasets
    col = lkf_collection([ds_a, ds_b])
    col.compute_density()
    assert hasattr(ds_a, 'density'), "density not cached on ds_a"
    assert hasattr(ds_b, 'density'), "density not cached on ds_b"


def test_collection_lkf_length_count(two_datasets):
    """Total segment count from collection equals sum of individual detected segments."""
    from lkf_tools.dataset import lkf_collection
    ds_a, ds_b = two_datasets
    col = lkf_collection([ds_a, ds_b])
    col.compute_lengths()

    n_a = sum(len(ds_a.lkfs[t]) for t in range(len(ds_a.lkfs)) if ds_a.lkfs[t] is not None)
    n_b = sum(len(ds_b.lkfs[t]) for t in range(len(ds_b.lkfs)) if ds_b.lkfs[t] is not None)
    assert len(col.lkf_length) == n_a + n_b


# ---------------------------------------------------------------------------
# Phase G step 6i: precondition guards
# ---------------------------------------------------------------------------

def test_compute_lengths_requires_detection(tmp_path):
    """compute_lengths() raises RuntimeError if detect_lkfs() was never called."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lkf_tools.dataset import process_dataset
    ds = process_dataset(str(NETCDF_FILE), output_path=str(tmp_path) + "/")
    with pytest.raises(RuntimeError, match="detect_lkfs"):
        ds.compute_lengths()


def test_compute_lifetime_requires_tracking(tmp_path):
    """compute_lifetime() raises RuntimeError if track_lkfs() was never called."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lkf_tools.dataset import process_dataset
    ds = process_dataset(str(NETCDF_FILE), output_path=str(tmp_path) + "/")
    ds.detect_lkfs(indexes=[0])
    with pytest.raises(RuntimeError, match="tracking"):
        ds.compute_lifetime()


def test_compute_growthrate_requires_tracking(tmp_path):
    """compute_growthrate() raises RuntimeError if track_lkfs() was never called."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from lkf_tools.dataset import process_dataset
    ds = process_dataset(str(NETCDF_FILE), output_path=str(tmp_path) + "/")
    ds.detect_lkfs(indexes=[0])
    with pytest.raises(RuntimeError, match="tracking"):
        ds.compute_growthrate()
