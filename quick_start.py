"""
QUICK START GUIDE - WESAD Stress Detection
==========================================

This guide walks you through running the stress detection pipeline
step-by-step with explanations.
"""

import sys
from pathlib import Path

def check_dataset():
    """Step 1: Check if dataset is available"""
    print("="*60)
    print("STEP 1: Checking Dataset")
    print("="*60)
    
    # Try to import config
    try:
        import config
        data_dir = config.get_data_dir()
        data_path = Path(data_dir)
    except ImportError:
        print("⚠️  config.py not found, using default paths")
        data_path = Path("C:/Users/louis/Downloads/WESAD/wesad_data")
    
    if not data_path.exists():
        print("❌ Dataset directory not found!")
        print(f"\n   Looking for: {data_path}")
        print("\n📥 OPTIONS:")
        print("1. Update the path in config.py:")
        print("   DATA_DIR = 'C:/Users/louis/Downloads/WESAD/wesad_data'")
        print("\n2. OR download dataset:")
        print("   Visit: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/")
        print("   Extract to the path shown above")
        return False
    
    # Check for subject files
    subjects_found = []
    for i in range(2, 18):
        subject_dir = data_path / f"S{i}"
        subject_file = subject_dir / f"S{i}.pkl"
        if subject_file.exists():
            subjects_found.append(i)
    
    if len(subjects_found) > 0:
        print(f"✓ Dataset found at: {data_path}")
        print(f"✓ Found {len(subjects_found)} subjects: {subjects_found}")
        return True
    else:
        print("❌ No subject files found in dataset directory")
        print("   Expected structure:")
        print("   wesad_data/")
        print("   ├── S2/S2.pkl")
        print("   ├── S3/S3.pkl")
        print("   └── ...")
        return False


def check_dependencies():
    """Step 2: Check if all dependencies are installed"""
    print("\n" + "="*60)
    print("STEP 2: Checking Dependencies")
    print("="*60)
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pywt': 'PyWavelets',
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n📦 To install:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def explore_one_subject():
    """Step 3: Explore data from one subject"""
    print("\n" + "="*60)
    print("STEP 3: Exploring One Subject's Data")
    print("="*60)
    
    try:
        import pickle
        import numpy as np
        
        # Try to get data directory from config
        try:
            import config
            data_dir = Path(config.get_data_dir())
        except ImportError:
            data_dir = Path("C:/Users/louis/Downloads/WESAD/wesad_data")
        
        data_path = data_dir / "S2" / "S2.pkl"
        
        if not data_path.exists():
            print(f"❌ S2 data not found at: {data_path}")
            print("   Skipping exploration.")
            return False
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print("\n📊 Subject S2 Data Structure:")
        print(f"  Main keys: {list(data.keys())}")
        print(f"  Signal keys: {list(data['signal'].keys())}")
        print(f"  Wrist sensors: {list(data['signal']['wrist'].keys())}")
        
        wrist = data['signal']['wrist']
        print("\n📈 Wrist Sensor Details:")
        for sensor, values in wrist.items():
            print(f"  {sensor}: shape={values.shape}")
        
        labels = data['label']
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n🏷️  Labels:")
        label_names = {0: "transient", 1: "baseline", 2: "stress", 
                      3: "amusement", 4: "meditation"}
        for label, count in zip(unique, counts):
            pct = count / len(labels) * 100
            print(f"  {label} ({label_names.get(label, 'unknown')}): {count} samples ({pct:.1f}%)")
        
        print("\n✓ Data exploration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error exploring data: {e}")
        return False


def run_feature_extraction_demo():
    """Step 4: Demo feature extraction on one subject"""
    print("\n" + "="*60)
    print("STEP 4: Feature Extraction Demo")
    print("="*60)
    
    try:
        import numpy as np
        from feature_extraction import WristFeatureExtractor, get_feature_names
        from ml_pipeline import WESADPipeline
        
        # Initialize
        pipeline = WESADPipeline()
        
        # Extract features for S2
        print("\n🔧 Extracting features for S2...")
        features, labels = pipeline.extract_features_for_subject(2)
        
        print(f"\n✓ Feature extraction complete!")
        print(f"  Number of windows: {len(features)}")
        print(f"  Feature dimensionality: {features.shape[1]}")
        print(f"  Stress windows: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
        print(f"  Non-stress windows: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
        
        # Show feature names
        feature_names = get_feature_names(WristFeatureExtractor())
        print(f"\n📋 First 10 features:")
        for i, name in enumerate(feature_names[:10], 1):
            print(f"  {i}. {name}")
        print(f"  ... and {len(feature_names) - 10} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_model():
    """Step 5: Train one model quickly on subset of subjects"""
    print("\n" + "="*60)
    print("STEP 5: Quick Model Training (3 subjects)")
    print("="*60)
    
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score
        from ml_pipeline import WESADPipeline
        
        pipeline = WESADPipeline()
        
        # Use only 3 subjects for quick demo
        test_subjects = [2, 3, 4]
        
        features_dict = {}
        labels_dict = {}
        
        print("\n🔧 Extracting features...")
        for subj in test_subjects:
            try:
                f, l = pipeline.extract_features_for_subject(subj)
                features_dict[subj] = f
                labels_dict[subj] = l
            except Exception as e:
                print(f"  Skipping S{subj}: {e}")
        
        if len(features_dict) < 2:
            print("❌ Not enough subjects for quick test")
            return False
        
        # Simple train-test split
        train_subjects = list(features_dict.keys())[:-1]
        test_subject = list(features_dict.keys())[-1]
        
        print(f"\n🎯 Training on subjects: {train_subjects}")
        print(f"   Testing on subject: {test_subject}")
        
        # Prepare data
        X_train = np.vstack([features_dict[s] for s in train_subjects])
        y_train = np.concatenate([labels_dict[s] for s in train_subjects])
        X_test = features_dict[test_subject]
        y_test = labels_dict[test_subject]
        
        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train
        print("\n🤖 Training Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        print(f"\n✓ Quick test complete!")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        print("\n💡 This is just a quick test with limited subjects.")
        print("   For full evaluation, run: python ml_pipeline.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in quick model training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete quick start guide"""
    print("\n" + "="*70)
    print(" "*15 + "WESAD STRESS DETECTION")
    print(" "*20 + "Quick Start Guide")
    print("="*70)
    
    # Run all steps
    steps = [
        ("Dataset Check", check_dataset),
        ("Dependencies Check", check_dependencies),
        ("Data Exploration", explore_one_subject),
        ("Feature Extraction Demo", run_feature_extraction_demo),
        ("Quick Model Training", run_quick_model),
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for step_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} - {step_name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL CHECKS PASSED!")
        print("="*70)
        print("\n📌 Next steps:")
        print("  1. Run full pipeline: python ml_pipeline.py")
        print("  2. Check outputs/ folder for results")
        print("  3. Review README.md for detailed documentation")
        print("\n🚀 Ready for deep learning? Ask me to proceed!")
    else:
        print("\n" + "="*70)
        print("⚠️  SOME CHECKS FAILED")
        print("="*70)
        print("\nPlease resolve the failed steps before proceeding.")
        print("Refer to README.md for troubleshooting help.")


if __name__ == "__main__":
    main()
