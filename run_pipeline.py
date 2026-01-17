"""
Quick Run Script - Start Full Pipeline
=======================================

Skips verification steps and jumps straight to training.
Use this once you've verified your setup works.
"""

import sys

# Try to import and configure
try:
    import config
    print(f"✓ Using data directory: {config.get_data_dir()}")
except ImportError:
    print("⚠️  config.py not found, using auto-detection")

print("\n" + "="*70)
print(" "*20 + "STARTING FULL PIPELINE")
print("="*70)
print("\nThis will:")
print("  1. Load all 15 subjects")
print("  2. Extract features (~70 per window)")
print("  3. Train 6 models with LOSO CV")
print("  4. Generate visualizations")
print("  5. Save results to outputs/")
print("\nExpected time: 10-30 minutes")
print("\nPress Ctrl+C to cancel...\n")

import time
time.sleep(2)

# Import and run main pipeline
from ml_pipeline import main

try:
    main()
except KeyboardInterrupt:
    print("\n\n⚠️  Pipeline interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n\n❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
