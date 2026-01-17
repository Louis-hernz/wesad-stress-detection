"""
Debug Script - Investigate WESAD Data Issues
=============================================

This script helps diagnose problems with feature extraction.
"""

import pickle
import numpy as np
from pathlib import Path
import sys

# Import config
try:
    import config
    DATA_DIR = config.get_data_dir()
except:
    DATA_DIR = "C:/Users/louis/Downloads/WESAD/wesad_data"

def debug_subject(subject_id=2):
    """
    Deep dive into one subject's data to find issues
    """
    print("="*70)
    print(f"DEBUGGING SUBJECT S{subject_id}")
    print("="*70)
    
    # Load data
    data_path = Path(DATA_DIR) / f"S{subject_id}" / f"S{subject_id}.pkl"
    
    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        return
    
    print(f"\n✓ Loading: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    print("\n" + "="*70)
    print("1. DATA STRUCTURE")
    print("="*70)
    
    # Check structure
    print(f"\nMain keys: {list(data.keys())}")
    print(f"Signal keys: {list(data['signal'].keys())}")
    
    if 'wrist' not in data['signal']:
        print("❌ No wrist data found!")
        return
    
    print(f"Wrist sensors: {list(data['signal']['wrist'].keys())}")
    
    # Check each sensor
    print("\n" + "="*70)
    print("2. SENSOR DATA SHAPES")
    print("="*70)
    
    wrist = data['signal']['wrist']
    
    for sensor, values in wrist.items():
        print(f"\n{sensor}:")
        print(f"  Type: {type(values)}")
        print(f"  Shape: {values.shape}")
        print(f"  Ndim: {values.ndim}")
        print(f"  Dtype: {values.dtype}")
        
        # Check for expected shapes
        if sensor == 'ACC':
            if values.ndim != 2 or (values.ndim == 2 and values.shape[1] != 3):
                print(f"  ⚠️  WARNING: Expected (n_samples, 3), got {values.shape}")
        else:
            if values.ndim != 1:
                print(f"  ⚠️  WARNING: Expected 1-D array, got {values.ndim}-D")
        
        print(f"  Min: {np.min(values):.3f}")
        print(f"  Max: {np.max(values):.3f}")
        print(f"  Mean: {np.mean(values):.3f}")
        print(f"  Has NaN: {np.any(np.isnan(values))}")
        print(f"  Has Inf: {np.any(np.isinf(values))}")
    
    # Check labels
    print("\n" + "="*70)
    print("3. LABEL DATA")
    print("="*70)
    
    labels = data['label']
    print(f"\nLabels:")
    print(f"  Shape: {labels.shape}")
    print(f"  Dtype: {labels.dtype}")
    print(f"  Min: {np.min(labels)}")
    print(f"  Max: {np.max(labels)}")
    
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "transient", 1: "baseline", 2: "stress", 
                  3: "amusement", 4: "meditation"}
    
    print(f"\nLabel distribution:")
    total = len(labels)
    for label, count in zip(unique, counts):
        pct = count / total * 100
        name = label_names.get(label, 'unknown')
        print(f"  {label} ({name:12s}): {count:6d} ({pct:5.1f}%)")
    
    # Check alignment
    print("\n" + "="*70)
    print("4. DATA ALIGNMENT CHECK")
    print("="*70)
    
    # Expected relationships
    chest_fs = 700  # Hz
    eda_fs = 4
    bvp_fs = 64
    temp_fs = 4
    acc_fs = 32
    
    print(f"\nExpected sampling rates:")
    print(f"  Labels (chest): {chest_fs} Hz")
    print(f"  EDA: {eda_fs} Hz")
    print(f"  BVP: {bvp_fs} Hz")
    print(f"  TEMP: {temp_fs} Hz")
    print(f"  ACC: {acc_fs} Hz")
    
    # Compute expected lengths
    n_labels = len(labels)
    expected_eda = n_labels // (chest_fs // eda_fs)
    expected_bvp = n_labels // (chest_fs // bvp_fs)
    
    print(f"\nExpected lengths (based on {n_labels} labels):")
    print(f"  EDA: ~{expected_eda} samples")
    print(f"  BVP: ~{expected_bvp} samples")
    
    actual_eda = len(wrist['EDA'])
    actual_bvp = len(wrist['BVP'])
    
    print(f"\nActual lengths:")
    print(f"  EDA: {actual_eda} samples")
    print(f"  BVP: {actual_bvp} samples")
    
    # Check if aligned
    eda_diff = abs(actual_eda - expected_eda) / expected_eda * 100
    bvp_diff = abs(actual_bvp - expected_bvp) / expected_bvp * 100
    
    print(f"\nAlignment error:")
    print(f"  EDA: {eda_diff:.1f}%")
    print(f"  BVP: {bvp_diff:.1f}%")
    
    if eda_diff > 10 or bvp_diff > 10:
        print("\n⚠️  WARNING: Significant alignment mismatch!")
        print("   This may cause issues with windowing.")
    
    # Test feature extraction
    print("\n" + "="*70)
    print("5. FEATURE EXTRACTION TEST")
    print("="*70)
    
    # Downsample labels to EDA rate
    downsample_factor = chest_fs // eda_fs
    labels_eda = labels[::downsample_factor][:actual_eda]
    
    print(f"\nDownsampled labels: {len(labels_eda)} samples")
    
    # Check for valid windows
    window_size = 240  # 60 seconds at 4 Hz
    window_shift = 1   # 0.25 seconds at 4 Hz
    
    n_possible_windows = (len(labels_eda) - window_size) // window_shift + 1
    print(f"Possible windows: {n_possible_windows}")
    
    # Count valid windows
    valid_windows = 0
    for i in range(n_possible_windows):
        start = i * window_shift
        end = start + window_size
        
        window_labels = labels_eda[start:end]
        valid_labels = window_labels[(window_labels > 0) & (window_labels != 4)]
        
        if len(valid_labels) >= len(window_labels) * 0.5:
            valid_windows += 1
    
    print(f"Valid windows (>50% non-transient): {valid_windows}")
    
    if valid_windows == 0:
        print("\n❌ PROBLEM FOUND: No valid windows!")
        print("\nReasons could be:")
        print("  - Too much transient (label 0) data")
        print("  - All data is meditation (label 4)")
        print("  - Labels are not properly aligned")
        
        # Show label sequence
        print(f"\nFirst 100 labels after downsampling:")
        print(labels_eda[:100])
    else:
        print(f"\n✓ Should extract ~{valid_windows} windows")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    issues = []
    
    # Check for issues
    if 'wrist' not in data['signal']:
        issues.append("Missing wrist data")
    
    for sensor in ['ACC', 'BVP', 'EDA', 'TEMP']:
        if sensor not in wrist:
            issues.append(f"Missing {sensor} sensor")
    
    if valid_windows == 0:
        issues.append("No valid windows found")
    
    if eda_diff > 10:
        issues.append(f"EDA alignment off by {eda_diff:.1f}%")
    
    if len(issues) > 0:
        print("\n❌ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ No major issues detected!")
        print(f"  Subject S{subject_id} should work fine.")


def check_all_subjects():
    """
    Quick check of all subjects
    """
    print("="*70)
    print("CHECKING ALL SUBJECTS")
    print("="*70)
    
    data_path = Path(DATA_DIR)
    
    summary = []
    
    for i in range(2, 18):
        subject_file = data_path / f"S{i}" / f"S{i}.pkl"
        
        if not subject_file.exists():
            summary.append((i, "MISSING", 0, 0))
            continue
        
        try:
            with open(subject_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            if 'wrist' not in data['signal']:
                summary.append((i, "NO_WRIST", 0, 0))
                continue
            
            labels = data['label']
            wrist = data['signal']['wrist']
            eda_len = len(wrist.get('EDA', []))
            
            # Quick valid window estimate
            chest_fs = 700
            eda_fs = 4
            labels_eda = labels[::chest_fs//eda_fs][:eda_len]
            
            # Count non-transient labels
            valid_labels = np.sum((labels_eda > 0) & (labels_eda != 4))
            valid_pct = valid_labels / len(labels_eda) * 100 if len(labels_eda) > 0 else 0
            
            # Estimate windows
            window_size = 240
            n_windows = max(0, (len(labels_eda) - window_size) // 1 + 1)
            
            status = "OK" if valid_pct > 30 and n_windows > 100 else "WARNING"
            
            summary.append((i, status, valid_pct, n_windows))
            
        except Exception as e:
            summary.append((i, f"ERROR: {e}", 0, 0))
    
    # Print summary
    print(f"\n{'Subject':<10} {'Status':<12} {'Valid %':<10} {'Est. Windows'}")
    print("-" * 70)
    
    for subj, status, valid_pct, n_windows in summary:
        print(f"S{subj:<9d} {status:<12s} {valid_pct:>6.1f}%   {n_windows:>8d}")
    
    # Count by status
    ok_count = sum(1 for _, s, _, _ in summary if s == "OK")
    warning_count = sum(1 for _, s, _, _ in summary if s == "WARNING")
    error_count = sum(1 for _, s, _, _ in summary if s.startswith("ERROR") or s in ["MISSING", "NO_WRIST"])
    
    print("\n" + "="*70)
    print(f"Total subjects: {len(summary)}")
    print(f"  OK: {ok_count}")
    print(f"  WARNING: {warning_count}")
    print(f"  ERROR/MISSING: {error_count}")
    
    if ok_count < 3:
        print("\n❌ Not enough subjects for training!")
        print("   Need at least 3 subjects with OK status.")
    else:
        print(f"\n✓ {ok_count} subjects should work for training.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Debug specific subject
        subject_id = int(sys.argv[1])
        debug_subject(subject_id)
    else:
        # Check all subjects
        check_all_subjects()
        
        print("\n" + "="*70)
        print("To debug a specific subject, run:")
        print("  python debug_data.py 2")
        print("="*70)
