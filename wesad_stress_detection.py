"""
WESAD Stress Detection - Enhanced Traditional ML Approach
========================================================

Project Structure:
1. Data Download & Exploration
2. Preprocessing & Feature Engineering
3. Traditional ML Baseline (reproduce paper)
4. Enhanced Feature Engineering
5. XGBoost/LightGBM Models
6. Evaluation & Comparison

Dataset Info:
- 15 subjects
- Wrist device: BVP (64Hz), EDA (4Hz), TEMP (4Hz), ACC (32Hz)
- Binary: Stress vs Non-Stress (baseline + amusement)
- Evaluation: LOSO Cross-Validation
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile

class WESADDataLoader:
    """Download and load WESAD dataset"""
    
    def __init__(self, data_dir=None):
        # Default paths to try (in order)
        if data_dir is None:
            possible_paths = [
                "C:/Users/louis/Downloads/WESAD/wesad_data",  # Your Windows path
                "/mnt/c/Users/louis/Downloads/WESAD/wesad_data",  # WSL path
                "./wesad_data",  # Current directory
                "../wesad_data",  # Parent directory
                "/home/claude/wesad_data"  # Original default
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_dir = path
                    print(f"✓ Found WESAD data at: {data_dir}")
                    break
            if data_dir is None:
                data_dir = possible_paths[0]  # Use first as default
        
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            print(f"⚠️  Warning: Data directory does not exist: {self.data_dir}")
            print(f"   You can specify custom path: WESADDataLoader(data_dir='your/path')")
        
    def download_dataset(self):
        """
        Download WESAD dataset from official source
        Note: The dataset is hosted at https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
        """
        print("=" * 60)
        print("WESAD Dataset Download")
        print("=" * 60)
        
        # The dataset needs to be manually downloaded due to terms of use
        # URL: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
        
        print("\n⚠️  MANUAL DOWNLOAD REQUIRED")
        print("\nPlease download the WESAD dataset manually:")
        print("1. Visit: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/")
        print("2. Accept terms and download WESAD.zip")
        print("3. Place it in:", self.data_dir)
        print("\nFor this tutorial, I'll create a data exploration script")
        print("that works once you have the dataset downloaded.")
        
        return False
    
    def load_subject_data(self, subject_id):
        """
        Load data for a single subject
        
        WESAD data structure (per subject):
        - subject/subjectX/subjectX.pkl contains:
          - 'signal': dict with 'chest' and 'wrist' data
          - 'label': array with condition labels
          - 'subject': subject info
        
        Labels:
        0 = not defined / transient
        1 = baseline
        2 = stress (TSST)
        3 = amusement
        4 = meditation
        """
        subject_file = self.data_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
        
        if not subject_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subject_file}")
        
        with open(subject_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        return data


def explore_data_structure():
    """
    Create a data exploration script to understand WESAD format
    """
    script = '''
# Data Exploration Template
# Run this once you have the WESAD dataset downloaded

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def explore_wesad_subject(subject_id=2):
    """Explore structure of WESAD data for one subject"""
    
    data_path = Path("/home/claude/wesad_data") / f"S{subject_id}" / f"S{subject_id}.pkl"
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    print("="*60)
    print(f"Subject S{subject_id} Data Structure")
    print("="*60)
    
    # Main keys
    print("\\nMain keys:", data.keys())
    
    # Signal structure
    print("\\nSignal keys:", data['signal'].keys())
    print("\\nWrist sensors:", data['signal']['wrist'].keys())
    
    # Wrist data details
    wrist = data['signal']['wrist']
    print("\\n" + "="*60)
    print("WRIST SENSOR DETAILS")
    print("="*60)
    
    for sensor, values in wrist.items():
        print(f"\\n{sensor}:")
        print(f"  Shape: {values.shape}")
        print(f"  Sample values: {values[:5].flatten()[:10]}")
    
    # Labels
    labels = data['label']
    print("\\n" + "="*60)
    print("LABELS")
    print("="*60)
    print(f"Shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Label counts:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = {0: "transient", 1: "baseline", 2: "stress", 
                     3: "amusement", 4: "meditation"}
        print(f"  {label} ({label_name.get(label, 'unknown')}): {count} samples")
    
    # Subject info
    print("\\n" + "="*60)
    print("SUBJECT INFO")
    print("="*60)
    if 'subject' in data:
        print(data['subject'])
    
    return data

# Run exploration
if __name__ == "__main__":
    data = explore_wesad_subject(2)
'''
    
    return script


if __name__ == "__main__":
    loader = WESADDataLoader()
    
    # Attempt to download (will show instructions)
    loader.download_dataset()
    
    # Generate exploration script
    print("\n" + "="*60)
    print("Generating data exploration template...")
    print("="*60)
    
    explore_script = explore_data_structure()
    
    with open("/home/claude/explore_wesad.py", "w") as f:
        f.write(explore_script)
    
    print("\n✓ Created exploration script: explore_wesad.py")
    print("\nNext steps:")
    print("1. Download WESAD dataset")
    print("2. Run: python explore_wesad.py")
    print("3. Proceed with preprocessing & feature extraction")
