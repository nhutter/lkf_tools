# Task: Parallelization

## Goal
Exploit the embarrassingly parallel structure of the detection and tracking
pipelines to reduce wall-clock time proportionally to available CPU cores.
All changes must produce bit-identical (or numerically equivalent within float
tolerance) output and pass the existing regression suite.

## Background
Profiling after Items 1–6 (see `task-performance.md`) shows:
- Detection: ~1.37s — each timestep index is fully independent
- Tracking: ~0.13s — each segment pair (`ilkf`) is fully independent

Both loops are embarrassingly parallel with no shared mutable state during
computation. Writes back to `self.*` must remain sequential.

## Implementation plan

### Item 9: Parallel detection across timesteps (`detect_lkfs`)

**Where:** `lkf_tools/dataset.py` — the `for it in [...]` loop in `detect_lkfs`.

**Approach:**
1. Extract the per-timestep computation (deformation rates → `eps_tot_list` →
   `lkf_detect_eps_multday` → lat/lon enrichment) into a private helper
   function `_detect_one(it)` that returns `(it, lkf, eps_tot_list)`.
2. Run `_detect_one` in parallel over all `it` values using
   `joblib.Parallel(n_jobs=-1, backend='loky')`.
3. Collect results and write back to `self.lkfs`, `self.timesteps_detected`,
   `self.data.lkfs`, and save `.npy` files sequentially.

**Caveats:**
- `self.data` xarray reads must be converted to numpy (`.values`) before
  spawning workers to avoid pickling the full Dataset.
- Write-back to `self.data.div/shr/vor` should be done in the sequential
  collection step.
- Test with `n_jobs=1` first to verify correctness, then benchmark with
  `n_jobs=-1`.

**Steps:**
- [ ] **Step 9a – Implement on feature branch** `perf/parallel-detection`
- [ ] **Step 9b – Run regression test** (`n_jobs=1` parity check first)
- [ ] **Step 9c – Benchmark** – report wall-clock before/after on sample input
      with `n_jobs=2` and `n_jobs=-1`
- [ ] **Step 9d – Code review checkpoint** – show diff, wait for go-ahead
- [ ] **Step 9e – Merge to `optimize`** only after explicit approval

---

### Item 10: Parallel tracking across segment pairs (`track_lkf`)

**Where:** `lkf_tools/tracking.py` — the `for ilkf, iseg_d in enumerate(lkf0_d)` loop.

**Approach:**
1. Extract the per-segment body (search area construction, ravel, set
   intersection, MHD check) into a helper `_track_one_segment(ilkf, iseg_d,
   lkf1, lkf1_sets, grid_shape, ...)` that returns a list of matched pairs.
2. Precompute `lkf1_sets = [set(np.ravel_multi_index(...)) for lkf1[i]]` once
   before the parallel loop to avoid redundant work inside workers.
3. Run with `joblib.Parallel(n_jobs=-1, backend='loky')` and flatten results.

**Caveats:**
- `lkf0_d`, `lkf1`, and grid arrays must be picklable (they are pure numpy —
  no issue expected).
- Overhead of spawning workers may outweigh gains for small inputs (few
  segments). Add a fallback: use sequential loop when
  `len(lkf0_d) < parallel_threshold` (suggest threshold = 50).

**Steps:**
- [ ] **Step 10a – Implement on feature branch** `perf/parallel-tracking`
- [ ] **Step 10b – Run regression test**
- [ ] **Step 10c – Benchmark** – report wall-clock before/after
- [ ] **Step 10d – Code review checkpoint** – show diff, wait for go-ahead
- [ ] **Step 10e – Merge to `optimize`** only after explicit approval

---

## Definition of done
- All regression tests pass
- Detection wall-clock time scales sub-linearly with number of timesteps on
  multi-core hardware
- No public API signatures changed
- `n_jobs` parameter (or equivalent) is configurable and defaults to `-1`
  (use all cores)
