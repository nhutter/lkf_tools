"""
Generate reference output for regression testing.

Runs detection (timesteps 0 and 1) and tracking (pair 0->1) on the McGill
sample file and saves the results to tests/fixtures/reference_output.npz.

Usage:
    conda run -n lkf_tools python tests/generate_reference.py

The script is intentionally idempotent: re-running it overwrites the fixture.
Detection/tracking output files are written to data/lkfs/McGill/ as usual.
"""

from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NETCDF_FILE = REPO_ROOT / "data" / "McGill" / "McGill_e2_1997_daily_means.nc"
OUTPUT_PATH = REPO_ROOT / "data" / "lkfs" / "McGill"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "reference_output.npz"

# Import after path setup so the package is found regardless of cwd
import sys
sys.path.insert(0, str(REPO_ROOT))
from lkf_tools.dataset import process_dataset

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("Initialising lkf_dataset …")
lkf_data = process_dataset(str(NETCDF_FILE), output_path=str(OUTPUT_PATH) + "/")

print("Detecting LKFs for timesteps 0 and 1 …")
lkf_data.detect_lkfs(indexes=[0, 1])

print("Tracking LKFs for pair 0 -> 1 …")
lkf_data.track_lkfs(indexes=[0], force_recompute=True)

# --- Load the saved .npy files to build the fixture ---
stem = NETCDF_FILE.stem  # "McGill_e2_1997_daily_means"

lkfs_0 = np.load(
    lkf_data.lkfpath / f"lkf_{stem}_001.npy", allow_pickle=True
)
lkfs_1 = np.load(
    lkf_data.lkfpath / f"lkf_{stem}_002.npy", allow_pickle=True
)

track_file = list(lkf_data.track_output_path.glob("lkf_tracked_pairs_*.npy"))
assert len(track_file) == 1, f"Expected exactly one tracked-pairs file, got {track_file}"
tracked_pairs = np.load(track_file[0], allow_pickle=True)

# Timestamps for the two detected steps
timestamps = np.array(
    lkf_data.time.values[[0, 1]], dtype="datetime64[ns]"
)

np.savez(
    FIXTURE_PATH,
    lkfs_0=lkfs_0,
    lkfs_1=lkfs_1,
    tracked_pairs=tracked_pairs,
    timestamps=timestamps,
)

print(f"Reference fixture written to {FIXTURE_PATH}")
print(f"  lkfs_0:        {len(lkfs_0)} segments")
print(f"  lkfs_1:        {len(lkfs_1)} segments")
print(f"  tracked_pairs: {tracked_pairs.shape}")
print(f"  timestamps:    {timestamps}")
