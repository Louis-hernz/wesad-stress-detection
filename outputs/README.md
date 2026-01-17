# WESAD Stress Detection - Enhanced ML Pipeline

## Project Overview

This project implements an **enhanced machine learning pipeline** for stress detection using the WESAD (Wearable Stress and Affect Detection) dataset. We use only **wrist-worn device data** to classify binary stress states (Stress vs Non-Stress).

### Goal
- **Match or beat** the paper's baseline of ~88% accuracy using wrist data
- Use **enhanced feature engineering** and modern ML algorithms
- Eventually progress to deep learning approaches

---

## Dataset Information

**WESAD Dataset:**
- **15 subjects** (S2-S17, excluding S1)
- **Wrist device (Empatica E4):**
  - BVP (Blood Volume Pulse): 64 Hz
  - EDA (Electrodermal Activity): 4 Hz
  - TEMP (Temperature): 4 Hz
  - ACC (3-axis Accelerometer): 32 Hz

**Binary Classification:**
- **Stress (1)**: TSST condition (Trier Social Stress Test)
- **Non-Stress (0)**: Baseline + Amusement conditions

**Evaluation:** Leave-One-Subject-Out (LOSO) Cross-Validation

---

## Project Structure

```
wesad_stress_detection/
│
├── wesad_stress_detection.py    # Main data loader
├── feature_extraction.py         # Enhanced feature extraction
├── ml_pipeline.py                # Complete ML pipeline
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── wesad_data/                   # Dataset directory (you create this)
│   ├── S2/
│   │   └── S2.pkl
│   ├── S3/
│   │   └── S3.pkl
│   └── ...
│
└── outputs/                      # Results (auto-created)
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── results_summary.csv
    └── detailed_results.txt
```

---

## Getting Started

### Step 1: Download the WESAD Dataset

1. Visit: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
2. Accept the terms of use
3. Download `WESAD.zip`
4. Extract to create a `wesad_data/` directory
5. Ensure structure is:
   ```
   wesad_data/
   ├── S2/
   │   └── S2.pkl
   ├── S3/
   │   └── S3.pkl
   ...
   ```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- pywt (PyWavelets)

### Step 3: Run the Pipeline

```bash
python ml_pipeline.py
```

This will:
1. Load data for all 15 subjects
2. Extract enhanced features (~70+ features per window)
3. Train 6 models with LOSO cross-validation
4. Generate visualizations and reports
5. Save results to `outputs/`

---

## Features Extracted

### From Paper (Baseline Features):
1. **ACC (19 features)**: Mean, std, integral, peak frequency for each axis + 3D magnitude
2. **BVP/Heart Rate (17 features)**: HR, HRV, NN50, RMSSD, frequency bands (ULF, LF, HF, UHF)
3. **EDA (17 features)**: Statistical, SCL/SCR decomposition, peak detection
4. **TEMP (6 features)**: Mean, std, min, max, range, slope

### Enhanced Features (Novel):
1. **Wavelet features**: Time-frequency decomposition for EDA
2. **Nonlinear HRV**: Poincaré plot features (SD1, SD2)
3. **Cross-modal features**: Correlations between BVP-EDA, EDA-TEMP
4. **Synchrony**: Peak alignment between cardiac and electrodermal activity
5. **Arousal index**: Combined measure of physiological arousal

**Total: ~70+ features**

---

## Models Implemented

### Baseline Models (from paper):
1. **Decision Tree**
2. **Random Forest** (100 trees)
3. **Linear Discriminant Analysis (LDA)**
4. **k-Nearest Neighbors (kNN, k=9)**

### Enhanced Models:
5. **XGBoost** (200 estimators, depth=6)
6. **LightGBM** (200 estimators, depth=6)

---

## Expected Results

**Paper Baseline (Wrist data):**
- Binary classification: ~88% accuracy
- F1-score: ~86%

**Our Target:**
- Match or exceed 88% accuracy
- Improved F1-score through enhanced features
- Better generalization through modern algorithms

---

## Outputs

After running the pipeline, check `outputs/`:

1. **`model_comparison.png`**: Bar charts comparing all models
2. **`confusion_matrix.png`**: Confusion matrix for best model
3. **`roc_curves.png`**: ROC curves with AUC scores
4. **`results_summary.csv`**: Summary table of all results
5. **`detailed_results.txt`**: Complete classification reports
6. **`feature_names.txt`**: List of all extracted features

---

## Understanding the Code

### Key Components:

#### 1. `WristFeatureExtractor` (feature_extraction.py)
- Extracts features using 60-second sliding windows (0.25s shift)
- Handles multi-rate sensors (4 Hz to 64 Hz)
- Implements both paper features + enhancements

#### 2. `WESADPipeline` (ml_pipeline.py)
- Loads and preprocesses all subjects
- Implements LOSO cross-validation
- Trains and evaluates all models
- Generates visualizations

#### 3. Key Methods:
```python
# Extract features for one subject
features, labels = pipeline.extract_features_for_subject(subject_id)

# Train with LOSO CV
results = pipeline.train_and_evaluate_model(model_name, model, features_dict, labels_dict)

# Run everything
results = pipeline.run_all_models(features_dict, labels_dict)
```

---

## Next Steps (Deep Learning)

Once we validate the ML baseline, we'll move to:

1. **1D CNN** for temporal patterns
2. **LSTM/GRU** for sequential dependencies
3. **Transformer** with attention mechanisms
4. **Multi-scale CNN** handling different sampling rates
5. **End-to-end learning** from raw signals

---

## Troubleshooting

### Issue: "Subject file not found"
- Ensure WESAD dataset is extracted to `wesad_data/`
- Check that subject folders are named `S2`, `S3`, etc.
- Verify `.pkl` files exist inside each folder

### Issue: "Not enough memory"
- Reduce number of subjects for testing
- Decrease window size or increase window shift
- Process subjects sequentially instead of loading all

### Issue: "Poor accuracy"
- Check class balance in your data
- Verify labels are correctly mapped (stress=1, non-stress=0)
- Try hyperparameter tuning
- Ensure feature extraction handles NaN/Inf values

---

## References

**Original Paper:**
Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). 
*Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* 
ICMI '18, October 16–20, 2018, Boulder, CO, USA.

**Dataset:** https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

---

## Contributing

Suggestions for improvement:
1. Feature selection using mutual information or SHAP
2. Hyperparameter optimization with Optuna
3. Ensemble methods combining multiple models
4. Person-specific calibration
5. Real-time implementation considerations

---

## Questions?

Feel free to ask about:
- Feature extraction details
- Model selection rationale
- Deep learning next steps
- Performance optimization
- Real-time deployment
