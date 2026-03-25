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

## Definition of done
- All regression tests pass
- End-to-end wall-clock time reduced by ≥ 20 % on sample input
- No public API signatures changed