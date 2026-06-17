# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
conda env create -f environment.yml
conda activate lkf_tools
python setup.py develop
```

## Architecture

The package detects and tracks Linear Kinematic Features (LKFs) — leads and pressure ridges — in gridded sea-ice deformation data.

### Core pipeline

The main entry point is `lkf_dataset` in `lkf_tools/dataset.py`. The typical workflow is:

1. Instantiate `lkf_dataset` with a NetCDF file containing ice velocity (`U`, `V`), concentration (`A`), and optionally pre-computed deformation rates (`div`, `shr`, `vor`).
2. Call `detect_lkfs()` — computes total deformation rate (`eps_tot = sqrt(div² + shr²)`), applies the image-processing pipeline from `detection.py`, and saves per-timestep `.npy` files under `<output_path>/<netcdf_stem>/`.
3. Call `track_lkfs()` — advects detected LKFs using drift data, computes Modified Hausdorff Distance between consecutive timesteps via `tracking.py`, and saves tracked pairs to `<output_path>/<netcdf_stem>/tracked_pairs/`.
4. Call `save()` to persist the whole `lkf_dataset` instance as a pickle. On re-instantiation with the same paths, the pickle is loaded automatically.

### Module responsibilities

- **`detection.py`** — Image processing pipeline: NaN-aware Gaussian filtering, histogram equalization, Difference-of-Gaussians (DoG) feature enhancement, skeletonization, segment extraction, and coordinate conversion to lat/lon. Key function: `lkf_detect_eps_multday()`.
- **`tracking.py`** — Advects LKF segments using sea-ice drift, then matches segments across timesteps using Modified Hausdorff Distance. Key functions: `drift_estimate()`, `track_lkf()`.
- **`rgps.py`** — Reads binary RGPS (RADARSAT Geophysical Processor System) files and provides the SSMI stereographic projection (`mSSMI()` via pyproj). Also contains helper functions `segs2latlon_model()` and `segs2epsvor()` used during detection.
- **`stats.py`** — Standalone statistics module with its own `lkf_dataset` class (different from `dataset.py`'s class) for loading already-generated `.npy` datasets and computing spatial/temporal statistics. Entry point: `load_lkf_dataset()`.
- **`_dir_filter.py`** — Low-level Cython/numpy routine `skeleton_along_max()` used inside the detection pipeline.

### Data formats

- LKFs are stored as numpy object arrays of segments, each segment a `(N_cols, N_points)` array. Columns are `[row_idx, col_idx, lon, lat, div, shr, vor]` (after lat/lon and deformation enrichment). With `compute_width=True` (default) two per-pixel half-width columns `halfw1, halfw2` are appended (indices 7,8). Projected `x, y` are then appended last, so the full layout is `[row, col, lon, lat, div, shr, vor, halfw1, halfw2, x, y]` and `indm0/indm1` point at `x, y` dynamically (9,10 with width, 7,8 without).
- Files are named `lkf_<stem>_<NNN>.npy` (detected) and `lkf_tracked_pairs_<stem>_to_<stem>.npy` (tracked pairs).
- The `stats.py` module reads these files from a directory tree organized by year subdirectories.

### Detection parameters (key ones)

| Parameter | Role |
|---|---|
| `max_kernel` / `min_kernel` | Gaussian kernel sizes for DoG filter |
| `dog_thres` | Threshold on DoG-filtered field |
| `eps_thres` | Minimum total deformation rate to keep a pixel |
| `lmin` | Minimum segment length (in grid cells) |
| `red_fac` | Spatial subsampling factor (resolution reduction) |
| `t_red` | Number of timesteps averaged before detection. Default `1` (single daily field). Values >1 average `t_red` days, but then the per-pixel LKF width is measured on the last day only, so width is precise only with `t_red=1`. |

All spatial parameters are automatically scaled by `corfac = 12.5km / mean_grid_spacing / red_fac` to stay resolution-independent.

### Legacy code

The `legacy_code/` directory contains older standalone scripts that predate the package structure. The `data/lemieux_example/` directory contains an example application for the CREG model grid. Neither is part of the installed package.

## Task files (load when working on a task)
- Performance improvements: @tasks/task-performance.md
- Parallelization:          @tasks/task-parallelization.md
- Stats integration:        @tasks/task-stats-integration.md
- Lemieux metrics:          @tasks/task-lemieux-metrics.md
```

And the directory layout becomes:
```
lkf_tools/
├── CLAUDE.md
└── tasks/
    ├── task-performance.md
    └── task-stats-integration.md
