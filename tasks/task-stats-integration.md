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
- [ ] **Step 6a - Repeats of `lkf_data.detect_lkfs`**: Check if `force_redetect=False` option surpresses
  repeated detection if detection was done before with saved output.
- [ ] **Step 6b - Implement `tqdm` progress bar**: Use `tqdm` isntead of printed progress statements
  in `lkf_dataset.detect_lkfs` (`Compute deformation rates and detect features for day *`); Implement
  progress bar for `lkf_dataset.track_lkfs` and all slow methods in `stats.py`. 
- [ ] **Step 6c - Solve index error in `detect_lkfs`**: Work on `task_bugfix-lkfs-index.md`.
- [ ] **Step 6d - Custom masks in `lkf_dataset`**: Implement optional input option for `lkf_dataset.mask`;
  check correct shape of input mask, if not raise warning and use current mask as default.
- [ ] **Step 6e - Double colorbar in `lkf_density.plot()`**: Remove second colorbar.
- [ ] **Step 6f - Default of `lkf_intersection`**: Change to `use_vorticity=True`.
- [ ] **Step 6g - Filter option in `lkf_intersection.plot()`**: Implement filter option to plot only newly
  formed LKFs (`lifetime==1`).
- [ ] **Step 6h - Store output of `stats.py` methods**: Save and reload the output all stats methods with 
  `compute` function; Implement a `force_recompute` option.
- [ ] **Step 6i - `stats.py` method without detected lkfs: Check if lkfs were detected or tracked before
  applying the 
- [ ] **Step 6i - Save date stamp in `lkf_dataset.detect_lkfs`**: Store a timestamp of the date with the
  LKF when saving it; add timestamp as additional column to the LKF
- [ ] **Step 6j - Comparative plotting for `lkf_collection`**: Implement `plot_type = ['flatten','compare']`
  option to either `flatten` the LKF property of all `lkf_datasets` and plot them as one dataset, or to 
  plot the LKF property of all `lkf_datasets` separately in the same figure to copmare.
- [ ] **Step 6k - Nested use of `lkf_collection`**: Implemented option to use `lkf_collection` instead of 
  `lkf_dataset` in the `lkf_collection.__init__()`. The option should allow to have a collection over 
  multiple years nested in a collection of different model. With options like `flatten` by year and `compare`
  for different models. Test this option with the `IFREMER` data in `tutorial_gen_dataset`.

---

## Definition of done
- All stat methods from `legacy_code/lkf_stats_tools.py` accessible via `lkf_dataset` instances  ✓
- `lkf_collection` covers multi-year aggregation  ✓
- Equivalence tests pass for every integrated function  ✓
- Standalone `stats.py` classes/functions preserved (backwards compatibility)  ✓
- Tutorial notebook runs end-to-end without errors  (Phase E step 4b pending)
