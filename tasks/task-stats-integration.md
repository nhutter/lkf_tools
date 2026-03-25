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

Work in batches of 2–4 related classes. For each batch:
- [ ] **3a – Tests** (equivalence vs legacy class on same data)
- [ ] **3b – Implement**
- [ ] **3c – Run tests**
- [ ] **3d – Docstrings**
- [ ] **3e – Review checkpoint**

Batch order (suggested):
1. `lkf_lengths` + `lkf_density`  ✓ done
2. `lkf_curvature` + `lkf_deformation`
3. `lkf_intersection` + `lkf_orientation`
4. `lkf_lifetime` + `lkf_growthrate`

---

## Phase C – Add stat methods to `lkf_dataset`

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

Each method: calls `compute_projected_coords()` if not already done, instantiates
stat class, caches result. Integrate alongside Phase B batches.

---

## Phase D – New `lkf_collection` class
- [ ] **3a – Tests**: construct from two `lkf_dataset` instances; verify aggregated
  lengths/density match manual concatenation.
- [ ] **3b – Implement** `lkf_collection` in `dataset.py`:
  - `__init__(self, datasets: list)` — list of `lkf_dataset` instances
  - Aggregation wrappers for each stat (iterate over `self.datasets`)
  - `load_lkf_dataset()` in `stats.py` updated to return `lkf_collection`
- [ ] **3c – Run tests**
- [ ] **3d – Docstrings**
- [ ] **3e – Review checkpoint**

---

## Phase E – Package exports & documentation
- [ ] **Step 4a – Update `__init__.py`**: export `lkf_dataset`, `lkf_collection`
- [ ] **Step 4b – Update README / tutorial notebook**: show new class-based API,
  replacing standalone function calls with equivalent `lkf_dataset` method calls.

---

## Definition of done
- All stat methods from `legacy_code/lkf_stats_tools.py` accessible via `lkf_dataset` instances
- `lkf_collection` covers multi-year aggregation
- Equivalence tests pass for every integrated function
- Standalone `stats.py` classes/functions preserved (backwards compatibility)
- Tutorial notebook runs end-to-end without errors
