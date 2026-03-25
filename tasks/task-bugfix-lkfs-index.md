# Bug: `detect_lkfs` writes `self.lkfs` out of bounds on last timestep batch

## Description

In `lkf_tools/dataset.py`, `detect_lkfs` stores the result of each detection
with:

```python
self.lkfs[it + itr] = lkf   # itr is the last value from range(self.t_red)
```

After the inner `for itr in range(self.t_red)` loop ends, `itr = t_red - 1`.
So the write index is `it + (t_red - 1)`.

The `lkfs` array is sized `ceil(time.size / t_red)`.  For the last valid outer
index `it_max = ceil(time.size / t_red) - 1`, the write target is:

```
it_max + (t_red - 1)  =  ceil(time.size / t_red) - 1 + t_red - 1
```

which exceeds `len(self.lkfs) - 1` whenever `t_red > 1`.

**Concrete example** — McGill 1997 file, `t_red=3`, 90 timesteps:
- `lkfs` size = 30 (indices 0–29)
- Last outer index: `it = 28` → write at `28 + 2 = 30` → `IndexError`
- `it = 27` → write at `27 + 2 = 29` → last valid slot

The `.npy` file for `it=28` is saved *before* the assignment (so the file
exists on disk), but `self.lkfs` and `self.timesteps_detected` are never
updated, and `lkf_data.save()` is never reached.

## Fix

Store the result at `it` (the outer loop index), not `it + itr`:

```python
self.lkfs[it] = lkf
self.timesteps_detected[it] = True
```

The inner loop index `itr` is an implementation detail of the multi-day
averaging; the logical timestep being detected is always `it`.

## Impact

- Detection silently fails for the last `ceil(t_red / 2)` outer indices when
  `t_red > 1`.
- The saved `.npy` file for the penultimate failing index exists on disk but
  is not registered in the in-memory dataset.
- `lkf_data.save()` (pickle) is never written after a full run, so every
  re-instantiation re-runs detection from scratch.

## Steps

- [ ] Fix `self.lkfs[it + itr] = lkf` → `self.lkfs[it] = lkf` in
  `dataset.py::detect_lkfs`
- [ ] Fix `self.timesteps_detected[it + itr]` → `self.timesteps_detected[it]`
  (same line group)
- [ ] Verify that the xarray bulk-write loop below uses `it` consistently
- [ ] Regenerate reference fixture (`tests/generate_reference.py`) if the
  stored index changes
- [ ] Run regression + stats integration tests
