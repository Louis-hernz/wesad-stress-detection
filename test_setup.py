"""
Test Script for Windows Setup
==============================

Quick test to verify your WESAD dataset is accessible.
"""

from pathlib import Path
import pickle
import sys

# Your WESAD data path (from the screenshot)
DATA_DIR = "C:/Users/Baseb/Downloads/WESAD/wesad_data"

def test_setup():
    """Test if everything is set up correctly"""

    print("="*70)
    print(" "*20 + "WESAD SETUP TEST")
    print("="*70)

    # Test 1: Check if directory exists
    print("\n[Test 1] Checking data directory...")
    data_path = Path(DATA_DIR)

    if not data_path.exists():
        print(f"❌ FAILED: Directory not found: {DATA_DIR}")
        print("\n💡 FIX:")
        print("   Update DATA_DIR in this file (test_setup.py) to match your path")
        return False

    print(f"✓ PASSED: Directory exists")

    # Test 2: Check for subject folders
    print("\n[Test 2] Checking for subject folders...")
    subjects_found = []

    for i in range(2, 18):
        subject_dir = data_path / f"S{i}"
        if subject_dir.exists() and subject_dir.is_dir():
            subjects_found.append(i)

    if len(subjects_found) == 0:
        print(f"❌ FAILED: No subject folders found")
        print(f"\n   Expected folders like: S2, S3, S4, ..., S17")
        print(f"   In directory: {DATA_DIR}")
        return False

    print(f"✓ PASSED: Found {len(subjects_found)} subject folders")
    print(f"   Subjects: {subjects_found}")

    # Test 3: Check for .pkl files
    print("\n[Test 3] Checking for subject data files...")
    pkl_files_found = []

    for subj in subjects_found:
        pkl_file = data_path / f"S{subj}" / f"S{subj}.pkl"
        if pkl_file.exists():
            pkl_files_found.append(subj)

    if len(pkl_files_found) == 0:
        print(f"❌ FAILED: No .pkl files found")
        print(f"\n   Expected files like: S2/S2.pkl, S3/S3.pkl, ...")
        return False

    print(f"✓ PASSED: Found {len(pkl_files_found)} .pkl files")
    print(f"   Files: {['S' + str(s) for s in pkl_files_found]}")

    # Test 4: Try to load one file
    print("\n[Test 4] Testing data loading...")
    test_subject = pkl_files_found[0]
    test_file = data_path / f"S{test_subject}" / f"S{test_subject}.pkl"

    try:
        with open(test_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        print(f"✓ PASSED: Successfully loaded S{test_subject}")

        # Check data structure
        print("\n[Test 5] Checking data structure...")

        required_keys = ['signal', 'label']
        for key in required_keys:
            if key not in data:
                print(f"❌ FAILED: Missing key '{key}' in data")
                return False

        if 'wrist' not in data['signal']:
            print(f"❌ FAILED: Missing 'wrist' in signal data")
            return False

        wrist_sensors = ['ACC', 'BVP', 'EDA', 'TEMP']
        missing_sensors = []
        for sensor in wrist_sensors:
            if sensor not in data['signal']['wrist']:
                missing_sensors.append(sensor)

        if missing_sensors:
            print(f"❌ FAILED: Missing wrist sensors: {missing_sensors}")
            return False

        print(f"✓ PASSED: Data structure is correct")

        # Print data info
        print("\n" + "="*70)
        print("DATA INFORMATION")
        print("="*70)

        wrist = data['signal']['wrist']
        print(f"\nWrist sensors found:")
        for sensor, values in wrist.items():
            print(f"  {sensor:8s}: shape={values.shape}")

        print(f"\nLabels: shape={data['label'].shape}")

        import numpy as np
        unique, counts = np.unique(data['label'], return_counts=True)
        label_names = {0: "transient", 1: "baseline", 2: "stress",
                      3: "amusement", 4: "meditation"}
        print(f"\nLabel distribution:")
        for label, count in zip(unique, counts):
            pct = count / len(data['label']) * 100
            print(f"  {label} ({label_names.get(label, 'unknown'):12s}): {count:6d} samples ({pct:5.1f}%)")

        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*70)

        print("\n🎉 Your setup is ready!")
        print("\nNext steps:")
        print("  1. Update config.py with your DATA_DIR path")
        print("  2. Run: python quick_start.py")
        print("  3. Run: python ml_pipeline.py")

        return True

    except Exception as e:
        print(f"❌ FAILED: Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🔍 Testing your WESAD setup...\n")

    success = test_setup()

    if not success:
        print("\n" + "="*70)
        print("⚠️  SETUP INCOMPLETE")
        print("="*70)
        print("\nPlease fix the issues above before running the pipeline.")
        sys.exit(1)

    sys.exit(0)
