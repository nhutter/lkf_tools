# Task: Performance Analysis & Speed-up

## Goal
Profile the detection and tracking pipeline and implement targeted speed-ups
that produce bit-identical (or numerically equivalent within float tolerance)
output on the same input data.

## Reference data contract
Before any optimisation work, we need a frozen reference output to diff against.
- [x] **Step 1 – Identify test input**
  Ask me to point to a small but realistic sample netCDF file (or create a
  synthetic one from the tutorial notebook). Target: runs in < 2 min on current
  code.
- [x] **Step 2 – Capture reference output**
  Run detection + tracking on the sample file. Save results to
  `tests/fixtures/reference_output.npz` (LKF pixel coordinates, track IDs,
  timestamps). Write the script as `tests/generate_reference.py`.
- [x] **Step 3 – Write regression test**
  Create `tests/test_regression.py` that loads the reference output and asserts
  equality (np.allclose for floats, exact match for integer indices/IDs).
  Make it runnable with `pytest tests/test_regression.py`.

## Profiling
- [x] **Step 4 – Profile detection**
  Use `cProfile` + `snakeviz` (or `line_profiler` on hotspot functions).
  Report top-10 slowest lines/functions. Do NOT change any code in this step.
- [x] **Step 5 – Profile tracking**
  Same as above for the tracking module. Report separately.
- [x] **Step 6 – Summarise findings**
  Present a prioritised table: function name | current time share | suggested
  optimisation (vectorise / numba / cache / algorithm change). Wait for my
  approval before proceeding.

## Optimisation (one optimisation at a time)

### Item 1: Vectorise ravel loops in `track_lkf` (branch: `perf/vectorise-ravel`)
- [x] **Step 7a – Implement on feature branch** `perf/vectorise-ravel`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – tracking 3.20s → 0.80s (75% reduction)
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 2: Bulk xarray LKF map writes in `dataset.py` (~2.20s saved)
- [x] **Step 7a – Implement on feature branch** `perf/bulk-lkf-map`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – detection ~5.2s → 2.97s
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 3: Replace `np.nansum` with `np.sum` for `nodetect_intm` (~1.55s saved)
- [x] **Step 7a – Implement on feature branch** `perf/nansum-to-sum`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – detection 2.97s → 1.55s (48% reduction)
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 4: Set intersection for `np.unique` overlap checks in `track_lkf` (~0.70s saved)
- [x] **Step 7a – Implement on feature branch** `perf/set-intersection`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – tracking 0.80s → 0.13s (84% reduction)
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 5: Cache `nansum_neighbours` result in restart block
- [x] **Step 7a – Implement on feature branch** `perf/cache-nansum`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – detection 1.55s → 1.45s
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 6: Move `nodetect_intm` copy outside inner loop
- [x] **Step 7a – Implement on feature branch** `perf/hoist-copy`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – detection 1.45s → 1.37s
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 7: Replace `sum_neighbours` 4D stride view with `scipy.ndimage.convolve` (~0.50s saved)
- [x] **Step 7a – Implement on feature branch** `perf/scipy-convolve`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – detection 1.37s → 0.99s (28% reduction)
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 8: Replace xarray `.where()` with numpy boolean masking in `detect_lkfs` (~0.07s saved)
- [x] **Step 7a – Implement on feature branch** `perf/numpy-mask`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – detection 0.99s → 0.97s
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

### Item 11: Deduplicate `search_area` via ravelled 1D indices in `track_lkf` (~0.01s saved)
- [x] **Step 7a – Implement on feature branch** `perf/ravel-dedup`
- [x] **Step 7b – Run regression test** – 6/6 pass
- [x] **Step 7c – Benchmark** – tracking 0.13s → 0.11s
- [x] **Step 7d – Code review checkpoint**
- [x] **Step 7e – Merged to `optimize`**

---

## Phase: Intersection angle speed-up (`lkf_intersection._process_timestep`)

### Profiling summary (McGill 28-day dataset, 28 timesteps × ~125 segments)

**Total `compute_intersection` time: ~0.80 s**

| Location | % time | Description |
|---|---|---|
| `L1087` `np.where(search_ind & (lkf_map != 0))` | **61.9 %** | Dense boolean grid AND + `np.where` scan over full ~400×400 grid per segment |
| `L1048` `lkf_poly_fit_p` → `np.polyfit` / `lstsq` | **16.4 %** | Full least-squares solve (×2 per intersection pair) just to fit a line to ≤20 points |
| `L1083–1085` `search_ind` nested loop | **9.8 %** | Python `for ix, iy` loops rebuilding a full boolean grid per segment |
| `L1090` `np.unique(lkf_map[ipoints])` | **3.0 %** | Sort-based unique on each segment's contact points |

**Root cause:** every segment scans the entire grid (60% of runtime). The actual intersection
neighbourhood is tiny (≤ a few dozen pixels), but the code allocates and scans a dense boolean
array of full grid size for each of the N segments.

---

### Proposed optimisations (prioritised)

#### Item 12: Sparse set-based neighbourhood search (~61 % saved)

**Branch:** `perf/intersection-sparse`

**Problem:** `search_ind` (dense bool, full grid) is created and ANDed with `lkf_map != 0` for
every segment.  The result has at most `N_pts × (2·dis_par+1)²` True cells — tiny compared with
the full grid.

**Fix – replace `lkf_map` + `search_ind` with a ravel-index dict:**
1. Precompute once:
   ```python
   pixel_to_segs = defaultdict(list)   # ravel_idx → [seg_ids]
   seg_ravel = []                       # list of sets, one per seg
   for iseg, seg in enumerate(segs):
       rv = np.ravel_multi_index([rows, cols], grid_shape)
       seg_ravel.append(set(rv))
       for r in rv:
           pixel_to_segs[r].append(iseg)
   ```
2. For each segment, compute its expanded ravel indices in one vectorised step
   (no Python `for ix, iy` loop):
   ```python
   dr = np.arange(-dis_par, dis_par + 1)
   offsets = (dr[:, None] * grid_shape[1] + dr[None, :]).ravel()
   base = np.ravel_multi_index(np.clip([rows_i, cols_i], ...), grid_shape)
   expanded = np.unique((base[:, None] + offsets).ravel())  # still fast, small array
   ```
3. Collect partner IDs directly from `pixel_to_segs` — no `np.where`, no `np.unique` on a grid.

**Bit-reproducibility:** same partner detection path, only the iteration order changes.
Sort `partners` before the angle loop to restore deterministic ordering.

**Correctness fix (bonus):** the current `lkf_map += iseg` accumulation is subtly broken when
≥2 segments share a pixel (the sum is mistaken for another segment index).  The dict approach
is exact.

**Expected saving: ~65 % of total intersection time.**

Steps:
- [x] **Step 12a – Implement on feature branch** `perf/intersection-sparse`
- [x] **Step 12b – Run regression test** – 48/48 pass
- [x] **Step 12c – Benchmark** – 0.797s → 0.161s (80% reduction)
- [x] **Step 12d – Code review checkpoint**
- [x] **Step 12e – Merged to `stats_integration`**

---

#### Item 13: Replace `np.polyfit` with direct slope formula (~16 % saved)

**Branch:** `perf/intersection-slope`

**Problem:** `lkf_poly_fit_p` calls `np.polyfit(x, y, 1)` which invokes the full `lstsq`
machinery (QR / SVD) on a ≤20-point array.  For a degree-1 fit the slope is a closed-form
ratio of dot products.

**Fix – replace `lkf_poly_fit_p` call in `_angle_near_point` with:**
```python
t = np.arange(x.size, dtype=float)
t -= t.mean()
px1 = np.dot(t, x) / np.dot(t, t)   # slope of x(t)
py1 = np.dot(t, y) / np.dot(t, t)   # slope of y(t)
p = py1 / px1
```
No matrix solve, no allocation of Vandermonde matrix, no SVD.  Result is numerically identical
to `np.polyfit(..., 1)[0]` for centred `t` (which `polyfit` does internally anyway).

**Bit-reproducibility:** should be bit-for-bit identical to the current polyfit slope because
numpy polyfit for deg=1 reduces to the same formula internally.  Verify with `np.testing.assert_allclose`.

**Expected saving: ~15 % of total intersection time.**

Steps:
- [x] **Step 13a – Implement on feature branch** `perf/intersection-slope`
- [x] **Step 13b – Run regression test** – 48/48 pass; max deviation vs polyfit: 8.88e-16 (machine ε)
- [x] **Step 13c – Benchmark** – Items 12+13 combined: 0.797s → 0.113s (86% reduction)
- [x] **Step 13d – Code review checkpoint**
- [x] **Step 13e – Merged to `stats_integration`**

---

## Definition of done
- All regression tests pass
- End-to-end wall-clock time reduced by ≥ 20 % on sample input
- No public API signatures changed