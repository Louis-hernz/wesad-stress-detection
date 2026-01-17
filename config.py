"""
Configuration file for WESAD Stress Detection Pipeline
========================================================

Edit the DATA_DIR path below to match your setup.
"""

import platform
from pathlib import Path

# ============================================================================
# MAIN CONFIGURATION - EDIT THIS!
# ============================================================================

# Your WESAD data directory path
# For Windows: Use forward slashes (/) or raw strings (r"C:\path\to\data")
DATA_DIR = "C:/Users/Baseb/Downloads/WESAD/wesad_data"

# Alternative: Uncomment the path that matches your setup
# DATA_DIR = "/mnt/c/Users/louis/Downloads/WESAD/wesad_data"  # WSL
# DATA_DIR = "./wesad_data"  # Current directory
# DATA_DIR = "/home/claude/wesad_data"  # Linux default

# ============================================================================
# AUTO-DETECTION (Don't edit unless you know what you're doing)
# ============================================================================

def get_data_dir():
    """
    Get the data directory, trying multiple locations
    """
    # First, try the configured path
    if Path(DATA_DIR).exists():
        return str(Path(DATA_DIR).resolve())

    # Try to auto-detect
    possible_paths = [
        DATA_DIR,
        "C:/Users/louis/Downloads/WESAD/wesad_data",
        "/mnt/c/Users/louis/Downloads/WESAD/wesad_data",
        "./wesad_data",
        "../wesad_data",
        "/home/claude/wesad_data",
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"✓ Found WESAD data at: {path}")
            return str(Path(path).resolve())

    print(f"⚠️  Warning: Could not find WESAD data directory!")
    print(f"   Configured path: {DATA_DIR}")
    print(f"   Please update DATA_DIR in config.py")
    return DATA_DIR


def check_dataset(data_dir=None):
    """
    Check if the dataset is properly set up
    """
    if data_dir is None:
        data_dir = get_data_dir()

    data_path = Path(data_dir)

    print("\n" + "="*60)
    print("WESAD DATASET CHECK")
    print("="*60)

    if not data_path.exists():
        print(f"❌ Directory does not exist: {data_path}")
        print("\nPlease:")
        print("1. Download WESAD from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/")
        print("2. Extract the dataset")
        print(f"3. Update DATA_DIR in config.py to point to your data")
        return False

    print(f"✓ Data directory exists: {data_path}")

    # Check for subject files
    subjects_found = []
    subjects_missing = []

    for i in range(2, 18):  # S2 to S17
        subject_dir = data_path / f"S{i}"
        subject_file = subject_dir / f"S{i}.pkl"

        if subject_file.exists():
            subjects_found.append(i)
        else:
            subjects_missing.append(i)

    print(f"\n✓ Found {len(subjects_found)} subjects: {subjects_found}")

    if subjects_missing:
        print(f"⚠️  Missing subjects: {subjects_missing}")

    if len(subjects_found) >= 10:
        print("\n✓✓ Dataset check PASSED! Ready to run pipeline.")
        return True
    else:
        print(f"\n❌ Only found {len(subjects_found)} subjects (need at least 10)")
        print("Please check your dataset extraction.")
        return False


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Where to save results
OUTPUT_DIR = "./outputs"

# Visualization settings
PLOT_DPI = 300
PLOT_FORMAT = "png"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Which subjects to use (None = all available)
SUBJECTS_TO_USE = None  # or list like [2, 3, 4, 5, 6]

# Feature extraction settings
WINDOW_SIZE_PHYSIO = 60  # seconds (from paper)
WINDOW_SIZE_ACC = 5      # seconds (from paper)
WINDOW_SHIFT = 0.25      # seconds (from paper)

# Model settings
RANDOM_STATE = 42
N_JOBS = -1  # Use all CPU cores

# XGBoost settings
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# LightGBM settings
LIGHTGBM_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

# Random Forest settings
RF_PARAMS = {
    'n_estimators': 100,
    'criterion': 'entropy',
    'min_samples_split': 20,
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS
}


if __name__ == "__main__":
    # Test configuration
    print("Testing WESAD configuration...")
    check_dataset()
