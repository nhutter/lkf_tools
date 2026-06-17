# Task: Integrate Lemieux et al. width & grid-angle metrics

## Goal
Port two metrics from the collaborator fork
(https://github.com/JFLemieux73/lkf_tools, paper
https://doi.org/10.5194/egusphere-2024-3831) into the `stats_integration` branch:

1. **LKF width** — per-pixel half-widths, computed during detection and stored
   on each segment like `div`/`shr`/`vor`.
2. **LKF-to-grid angle** — angle of the LKF relative to the model grid axes,
   merged into the intersection-angle computation and evaluated at each
   intersection point.

The other overlapping metrics (length, density, intersection detection/angle,
orientation) are **not** changed — see "Out of scope".

## Approved design decisions
- **Width storage:** two half-widths (`halfw1`, `halfw2`), in **grid cells**,
  bit-faithful to Lemieux's `lkf_calc_width`.
- **Width perpendicular direction:** Lemieux's 8-direction tangent
  quantization (`NaN` where `sdelt ∉ {1,2,4}`). Faithful, not the continuous fit.
- **Width default:** **ON by default** in detection. This changes the saved
  `.npy` column count for all runs → regression fixture must be regenerated
  (see Phase A, step A4).
- **Grid-angle location/frame:** per-intersection, in the **projected** frame
  (reuse `self.proj`; Option B). Computed at each intersection contact point.
- **Grid-angle on legacy datasets:** the old nested `stats.lkf_dataset`
  structure lacks projection info → **skip silently** (store `None`).

## Reference implementation (collaborator code, for reuse)
- `lkf_tools/lkf_metrics.py::lkf_calc_width` (L146) — width inner loop to port.
- `lkf_tools/lkf_metrics.py::lkf_angles_with_grid` (L1038) + `xy_coor` (L583) +
  `calc_int_angle` (L458) — grid-angle reference (we adapt frame to projected).
- Note: Lemieux uses `ishift/jshift` because his detection crops the domain and
  reopens the netCDF. **We do not need shifts** — segment `row/col` index
  directly into the cropped+subsampled `eps_tot`/lon/lat already used in
  detection (`dataset.py:301`, `dataset.py:327-338`).

## Current-package anchor points
- `detection.py::segs2epsvor` (L992) — template for the new `segs2width`.
- `dataset.py:332` — `segs2epsvor` call site; insert width call right after,
  reusing the cropped/subsampled `eps_tot` from `dataset.py:301`.
- `dataset.py::compute_projected_coords` — currently hardcodes `indm0=7,
  indm1=8`; must become dynamic.
- `stats.py::lkf_intersection._process_timestep` / `_angle_near_point`
  (L1032/L1071) — host for the grid-angle.
- Column layout table in `CLAUDE.md` — must be updated.

---

## Phase A – Width metric

### Column layout change (width ON by default)
New per-segment layout after detection:
`[row, col, lon, lat, div, shr, vor, halfw1, halfw2]` (indices 0–8).
After `compute_projected_coords`: append `x, y` at 9, 10.
Index pointers: `self.indwidth0=7`, `self.indwidth1=8`, dynamic
`self.indm0=ncols-2`, `self.indm1=ncols-1`.

**Branch:** `feat/lkf-width`

- [x] **A1 – Implement `segs2width`** in `detection.py`
  - Signature `segs2width(segs, eps_tot, dsearch=5, frac=0.5)`.
  - Port `lkf_calc_width` inner loop verbatim: central-difference tangent
    (one-sided at ends) → `av,bv` quantization → `sdelt ∈ {1,2,4}` guard →
    `target = frac*eps_tot[seg pixel]` → walk `s=1..dsearch` along `v1`/`v2`,
    first cell below `target` sets `halfw`; capped at `dsearch` distance if
    never below. `NaN` outside the guard.
  - Drop `ishift/jshift`; index `eps_tot[seg[0], seg[1]]` directly.
  - Append `halfw1`, `halfw2` as two columns (cells).
- [x] **A2 – Wire into detection** in `dataset.py`
  - Call `segs2width` right after `segs2epsvor` (`dataset.py:332`), reusing the
    cropped/subsampled `eps_tot` (`dataset.py:301`).
  - Add params `width_dsearch=5`, `width_frac=0.5` (and `compute_width=True`).
  - Make `compute_projected_coords` set `indm0/indm1` from array width; add
    `self.indwidth0/1`.
- [x] **A3 – Update layout docs** — column table in `CLAUDE.md` and the
  `task-stats-integration.md` "Column layout after projection" note.
- [x] **A4 – Regenerate regression fixture** — rerun `tests/generate_reference.py`
  so `reference_output.npz` reflects the new 9/11-column layout; update the
  column-count assertions in `tests/test_stats_integration.py`.
- [x] **A5 – New width test** — `test_lemieux_metrics.py::test_width`: assert two
  width columns appended; spot-check halfw values against a hand-computed walk
  on a synthetic straight LKF in a known `eps_tot` field.
- [x] **A6 – Run full test suite** — regression + stats-integration + new tests.
- [x] **A7 – Docstrings**
- [x] **A8 – Review checkpoint** — show diff, wait for go-ahead.

---

## Phase B – LKF-to-grid angle

**Branch:** `feat/lkf-grid-angle`

- [x] **B1 – Projected grid axes** — in `lkf_intersection`, build `X,Y` arrays
  once per timestep by projecting the cropped lon/lat grid
  (`self.lon/self.lat` sliced by `index_x/index_y/red_fac`) with `self.proj`.
  Pass grid + proj through `__init__` → `_process_timestep`.
- [x] **B2 – Grid-axis angle at contact pixel** — central-difference `X,Y` at
  `(r,c)` to get grid-i axis `(X[r,c+1]−X[r,c−1], …)` and grid-j axis (row
  step); convert to angles in projected space.
- [x] **B3 – Per-intersection grid angle** — for each of the two participating
  segments: `grid_angle = min(|ang_seg − ang_gridi|, |ang_seg − ang_gridj|)`
  folded to nearest axis (matches Lemieux `min(anglex,angley)`). Store in new
  parallel list `self.lkf_interc_gridangle` (per-segment, like his
  `xangle1/xangle2`).
- [x] **B4 – Legacy path** — old nested `stats.lkf_dataset` (no `self.proj`):
  skip silently, store `None`.
- [x] **B5 – Plot** — opt-in histogram of grid-angles alongside the existing
  intersection-angle `plot()`.
- [x] **B6 – Test** — `test_lemieux_metrics.py::test_grid_angle`: synthetic
  LKF aligned with a grid axis → grid-angle ≈ 0; 45°-rotated → ≈ 45. Legacy
  dataset → `None`.
- [x] **B7 – Run full test suite**
- [x] **B8 – Docstrings**
- [x] **B9 – Review checkpoint** — show diff, wait for go-ahead.

---

## Out of scope (intentionally not changed)
The overlapping metrics differ in algorithm/convention from Lemieux but the
existing package versions are kept:
- **Intersection detection/angle** — package uses sparse pixel-adjacency +
  local slope fit; Lemieux uses Shapely `LineString` crossing, endpoint
  extrapolation, X/T/Y classification, and a signed quadrant-aware conjugate
  angle. His X/T/Y + conjugate convention could be added later as opt-in.
- **Length** — package: Euclidean in projected (x,y); Lemieux: pixel-to-pixel
  haversine on the sphere.
- **Density** — package: histogram in projected (x,y) + ice mask + cartopy;
  Lemieux: accumulate onto the native model grid.
- **Orientation** — package: geographic orientation; Lemieux's only angle is
  the grid-relative one (added in Phase B). Complementary, not duplicates.

## Definition of done
- Width columns present on every detected segment by default; faithful to
  Lemieux (2 half-widths, cells, 8-dir quantization).
- Grid-angle available per intersection in `lkf_intersection`; silently skipped
  for legacy datasets.
- Regression fixture regenerated; all tests pass.
- No public API signatures broken (new params are keyword-only with defaults).
- `CLAUDE.md` column-layout table updated.
