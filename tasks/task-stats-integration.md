# Task: Integrate Statistical Tools into lkf_dataset

## Goal
Unify the two existing `lkf_dataset` classes (detection/tracking in `dataset.py`; statistics in
`stats.py`) into a **single class** that handles one netCDF file end-to-end. Multi-year analysis
moves to a new `lkf_collection` class. All stat classes from `legacy_code/lkf_stats_tools.py`
are migrated to `stats.py` as helpers and exposed as methods on `lkf_dataset`.

## Architecture decision (approved)
- **Option A**: extend `dataset.py::lkf_dataset` with statistics methods.
- Projected (x, y) coordinates computed from lat/lon via `pyproj.Proj`.
- `lkf_dataset` covers one netCDF file: detect + track + stats.
- `lkf_collection` covers multi-year / multi-file aggregation and comparison.
- Standalone `stats.py` functions/classes preserved for backwards compatibility.

### Target layout
```
lkf_tools/
├── dataset.py  → lkf_dataset   (single netCDF: detect + track + stats)
│               → lkf_collection (multi-year: aggregates lkf_dataset instances)
└── stats.py    → stat helper classes (lkf_lengths, lkf_density, …)
                → lkf_poly_fit, power_law_fit, runmean, ks   (back-compat)
                → load_lkf_dataset                            (back-compat wrapper)
```

### Column layout after projection
`[row, col, lon, lat, div, shr, vor, x, y]`  — indices 7,8 are projected metres.
Index pointers stored as `self.indm0=7, self.indm1=8`.

---

## Audit phase
- [x] **Step 1 – Inventory lkf_stats_tools.py**
  Listed every public function/class with inputs, outputs, description.
- [x] **Step 2 – Draft refactoring plan**
  Plan presented and approved (see Architecture decision above).

---

## Phase A – Coordinate projection  *(prerequisite for all stats)*
- [x] **3a – Tests**: `test_stats_integration.py::test_projected_coords` —
  assert `self.lkfs[t]` has 9 columns after `compute_projected_coords()`; spot-check
  x,y values against manual `pyproj.Proj` call on the same lon,lat.
- [x] **3b – Implement** `lkf_dataset.compute_projected_coords(proj=None)`:
  appends x,y columns; sets `self.proj`, `self.indm0=7`, `self.indm1=8`.
- [x] **3c – Run tests** – 5/5 pass; 6/6 regression tests still pass
- [x] **3d – Docstring**
- [x] **3e – Review checkpoint**

---

## Phase B – Migrate stat classes to `stats.py`
Migrate from `legacy_code/lkf_stats_tools.py` into `lkf_tools/stats.py`:
`lkf_lengths`, `lkf_density`, `lkf_curvature`, `lkf_deformation`,
`lkf_intersection`, `lkf_orientation`, `lkf_lifetime`, `lkf_growthrate`,
`ks`, `runmean`, `power_law_fit`.

Adapt internal loops: flat timestep array (`lkf_dataset.lkfs[t]`) instead of
nested year/day structure. Classes detect old vs new structure via
`hasattr(lkf, 'lkf_dataset')`.

Batch order:
1. `lkf_lengths` + `lkf_density`  ✓ done
2. `lkf_curvature` + `lkf_deformation`  ✓ done
3. `lkf_intersection` + `lkf_orientation`  ✓ done (with `use_vorticity` support)
4. `lkf_lifetime` + `lkf_growthrate`  ✓ done

Also added `mask_instead=False` to `compute_MHD_segment` in `tracking.py` (needed by
`lkf_growthrate`). `t_red > 1` index mapping fixed in `lkf_lifetime` and `lkf_growthrate`.

---

## Phase C – Add stat methods to `lkf_dataset`  ✓ done (integrated with Phase B)

| Method | Stat class | Cached on |
|---|---|---|
| `compute_lengths()` | `lkf_lengths` | `self.lengths` |
| `compute_density(grid_res)` | `lkf_density` | `self.density` |
| `compute_curvature()` | `lkf_curvature` | `self.curvature` |
| `compute_deformation()` | `lkf_deformation` | `self.deformation` |
| `compute_intersection()` | `lkf_intersection` | `self.intersection` |
| `compute_orientation()` | `lkf_orientation` | `self.orientation` |
| `compute_lifetime()` | `lkf_lifetime` | `self.lifetime` |
| `compute_growthrate()` | `lkf_growthrate` | `self.growthrate` |

---

## Phase D – New `lkf_collection` class
- [x] **3a – Tests**: 5 tests in `test_stats_integration.py` — construct, compute_lengths,
  flat array equality, compute_density, segment count.
- [x] **3b – Implement** `lkf_collection` in `dataset.py`:
  - `__init__(self, datasets: list)` — list of `lkf_dataset` instances
  - Aggregation wrappers for all stat methods (iterate over `self.datasets`)
  - Flat properties: `lkf_length`, `lkf_curvature`, `lkf_orientation`, `lkf_lifetime`,
    `lkf_deformation`, `lkf_interc`
  - `load_lkf_dataset()` in `stats.py` updated with docstring pointing to new API
- [x] **3c – Run tests** – 48/48 pass
- [x] **3d – Docstrings**
- [x] **3e – Review checkpoint**

---

## Phase E – Add plot functions for stat methods to `lkf_dataset`
Add plotting routines for all statistical methods to be called with e.g. 
`lkf_dataset.lkf_density.plot()`. Style of the plots should follow the design in 
https://doi.org/10.1029/2021jc017666 or https://doi.org/10.5194/tc-14-93-2020
- [x] **Step 4a – Implement**: plotting routines in `stats.py` for all 8 stat classes.
  Feedback applied: curvature → scatter, deformation → LogNorm, density → cartopy + ice mask.
- [x] **Step 4b – Inherent routines in `lkf_collection`**: `plot_lengths`, `plot_density`,
  `plot_curvature`, `plot_deformation`, `plot_intersection`, `plot_orientation`, `plot_lifetime`,
  `plot_growthrate` added to `lkf_collection`; all accept `by_dataset=True` for per-dataset overlay.

---

## Phase F – Package exports & documentation
- [x] **Step 5a – Update `__init__.py`**: export `lkf_dataset`, `lkf_collection`,
  `process_dataset`
- [x] **Step 5b – Update README / tutorial notebook**: 28-cell statistics section added to
  `notebooks/tutorial_gen_dataset.ipynb` covering `compute_*()`, all 8 `plot()` methods,
  and `lkf_collection` API.

---

## Phase G - Small changes
- [x] **Step 6a - Repeats of `lkf_data.detect_lkfs`**: `force_redetect=False` (default) now skips
  timesteps whose `.npy` files already exist in `lkfpath`.
- [x] **Step 6b - Implement `tqdm` progress bar**: `tqdm` replaces print statements in
  `detect_lkfs` and `track_lkfs`; falls back gracefully if tqdm is not installed.
- [x] **Step 6c - Solve index error in `detect_lkfs`**: Fixed in a prior session (bugfix commit).
- [x] **Step 6d - Custom masks in `lkf_dataset`**: `mask=None` parameter added to `__init__`;
  shape mismatch raises a warning and falls back to the default mask.
- [x] **Step 6e - Double colorbar in `lkf_density.plot()`**: Internal colorbar removed; caller
  is responsible for adding one (as shown in the tutorial notebook).
- [x] **Step 6f - Default of `lkf_intersection`**: Changed to `use_vorticity=True`.
- [x] **Step 6g - Filter option in `lkf_intersection.plot()`**: `lifetime=None` parameter added;
  when set, only pairs where both segments have `lifetime == 1` are plotted.
- [x] **Step 6h - Store output of `stats.py` methods**: All `compute_*` methods save results as
  `stats_<name>.pkl` in `lkfpath`; `force_recompute=False` reloads from disk on subsequent calls.
- [x] **Step 6i - `stats.py` method without detected lkfs**: `_require_detection()` and
  `_require_tracking()` helpers added; all 8 `compute_*` methods raise `RuntimeError` with a
  clear message if the prerequisite step has not been run.
- [x] **Step 6i - Save date stamp in `lkf_dataset.detect_lkfs`**: `self.timestep_dates` dict
  (timestep index → `numpy.datetime64`) is populated during detection.
- [x] **Step 6j - Comparative plotting for `lkf_collection`**: All `plot_*` methods accept
  `by_dataset=True/False`; dataset labels shown via legend; `plot_density(by_dataset=True)` uses
  one subplot per dataset.
- [x] **Step 6k - Nested use of `lkf_collection`**: `lkf_collection.__init__` accepts other
  `lkf_collection` instances in `datasets`; each level has its own `by_dataset` flag and `label`;
  `plot_*` methods recurse correctly through nested collections.

---

## Definition of done
- All stat methods from `legacy_code/lkf_stats_tools.py` accessible via `lkf_dataset` instances  ✓
- `lkf_collection` covers multi-year aggregation  ✓
- Equivalence tests pass for every integrated function  ✓
- Standalone `stats.py` classes/functions preserved (backwards compatibility)  ✓
- Tutorial notebook runs end-to-end without errors  (Phase E step 4b pending)
