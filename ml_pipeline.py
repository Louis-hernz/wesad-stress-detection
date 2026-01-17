"""
Complete ML Pipeline for WESAD Stress Detection
================================================

Pipeline:
1. Load and preprocess data
2. Extract features
3. Train models (Baseline + Enhanced)
4. Evaluate with LOSO CV
5. Compare results
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve)
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

from feature_extraction import WristFeatureExtractor, get_feature_names


class WESADPipeline:
    """
    Complete pipeline for WESAD stress detection
    """
    
    def __init__(self, data_dir=None):
        # Auto-detect data directory or use provided path
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
                    print(f"✓ Auto-detected WESAD data at: {data_dir}")
                    break
            if data_dir is None:
                data_dir = possible_paths[0]  # Default to first option
                print(f"⚠️  Data directory not found, using: {data_dir}")
        
        self.data_dir = Path(data_dir)
        self.subjects = list(range(2, 18))  # S2 to S17 (15 subjects)
        self.feature_extractor = WristFeatureExtractor()
        self.scaler = StandardScaler()
        
        # Label mapping
        self.label_map = {
            1: 0,  # baseline -> non-stress
            2: 1,  # stress -> stress
            3: 0,  # amusement -> non-stress
        }
    
    def load_subject(self, subject_id: int) -> Dict:
        """Load data for one subject"""
        subject_file = self.data_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
        
        with open(subject_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        return data
    
    def preprocess_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert labels to binary stress/non-stress
        0 (transient) -> -1 (to be filtered)
        1 (baseline) -> 0 (non-stress)
        2 (stress) -> 1 (stress)
        3 (amusement) -> 0 (non-stress)
        4 (meditation) -> -1 (to be filtered)
        """
        new_labels = np.full_like(labels, -1)
        
        for old_label, new_label in self.label_map.items():
            new_labels[labels == old_label] = new_label
        
        return new_labels
    
    def extract_features_for_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract all features for one subject
        
        Returns:
            features: (n_windows, n_features)
            labels: (n_windows,) binary labels
        """
        print(f"Processing S{subject_id}...", end=" ")
        
        try:
            # Load data
            data = self.load_subject(subject_id)
            
            # Get wrist sensors
            wrist_data = data['signal']['wrist']
            
            # Validate sensor presence
            required_sensors = ['ACC', 'BVP', 'EDA', 'TEMP']
            missing = [s for s in required_sensors if s not in wrist_data]
            if missing:
                print(f"SKIP (missing sensors: {missing})")
                return np.array([]), np.array([])
            
            # Get labels and resample to match EDA (lowest frequency)
            labels = data['label']
            
            # Validate data shapes and log any issues
            try:
                acc_shape = wrist_data['ACC'].shape
                bvp_shape = wrist_data['BVP'].shape
                eda_shape = wrist_data['EDA'].shape
                temp_shape = wrist_data['TEMP'].shape
                
                # Ensure 1-D for BVP, EDA, TEMP
                if len(bvp_shape) > 1:
                    wrist_data['BVP'] = wrist_data['BVP'].flatten()
                if len(eda_shape) > 1:
                    wrist_data['EDA'] = wrist_data['EDA'].flatten()
                if len(temp_shape) > 1:
                    wrist_data['TEMP'] = wrist_data['TEMP'].flatten()
                
                # Ensure ACC is 2-D (n_samples, 3)
                if wrist_data['ACC'].ndim == 1:
                    acc_1d = wrist_data['ACC']
                    wrist_data['ACC'] = np.column_stack([acc_1d, np.zeros_like(acc_1d), np.zeros_like(acc_1d)])
                elif wrist_data['ACC'].ndim > 2:
                    wrist_data['ACC'] = wrist_data['ACC'].reshape(-1, 3)
                    
            except Exception as e:
                print(f"SKIP (shape validation error: {e})")
                return np.array([]), np.array([])
            
            # Validate data shapes
            if len(labels) == 0:
                print(f"SKIP (no labels)")
                return np.array([]), np.array([])
            
            # Resample labels to EDA frequency (4 Hz)
            # Labels are typically at chest sensor frequency (700 Hz)
            chest_fs = 700
            eda_fs = 4
            downsample_factor = chest_fs // eda_fs
            labels_eda = labels[::downsample_factor][:len(wrist_data['EDA'])]
            
            # Ensure all signals are same length
            min_len = min(len(wrist_data['EDA']), 
                         len(wrist_data['TEMP']), 
                         len(labels_eda))
            
            # Check if we have enough data
            if min_len < 240:  # Less than 60 seconds at 4 Hz
                print(f"SKIP (insufficient data: {min_len} samples)")
                return np.array([]), np.array([])
            
            wrist_data_aligned = {
                'ACC': wrist_data['ACC'][:min_len * 8],  # 32 Hz
                'BVP': wrist_data['BVP'][:min_len * 16],  # 64 Hz
                'EDA': wrist_data['EDA'][:min_len],  # 4 Hz
                'TEMP': wrist_data['TEMP'][:min_len],  # 4 Hz
            }
            labels_aligned = labels_eda[:min_len]
            
            # DON'T preprocess labels here - pass original labels to feature extraction
            # Feature extraction will filter transient/meditation
            # Then we map the returned labels to binary
            
            # Extract features (pass ORIGINAL labels, not preprocessed)
            features, feature_labels = self.feature_extractor.extract_all_features(
                wrist_data_aligned, labels_aligned  # Original labels: 0,1,2,3,4
            )
            
            # Check if we got any windows
            if len(features) == 0:
                print(f"SKIP (no valid windows extracted)")
                return np.array([]), np.array([])
            
            # feature_labels contains the original labels (1, 2, or 3) that passed filtering
            # Convert to binary (0/1) using the label_map
            binary_labels = np.array([self.label_map.get(l, -1) for l in feature_labels])
            
            # Filter out any invalid labels (should be rare now)
            valid_mask = binary_labels >= 0
            features = features[valid_mask]
            binary_labels = binary_labels[valid_mask]
            
            if len(features) == 0:
                print(f"SKIP (all windows filtered out)")
                return np.array([]), np.array([])
            
            # Count classes
            unique_classes = np.unique(binary_labels)
            if len(unique_classes) < 2:
                print(f"SKIP (only {len(unique_classes)} class: {unique_classes})")
                return np.array([]), np.array([])
            
            n_stress = np.sum(binary_labels == 1)
            n_nonstress = np.sum(binary_labels == 0)
            
            print(f"✓ {len(features)} windows (stress:{n_stress}, non-stress:{n_nonstress})")
            
            return features, binary_labels
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])
    
    def load_all_data(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Load and extract features for all subjects
        
        Returns:
            features_dict: {subject_id: features}
            labels_dict: {subject_id: labels}
        """
        features_dict = {}
        labels_dict = {}
        
        print("\n" + "="*60)
        print("EXTRACTING FEATURES FOR ALL SUBJECTS")
        print("="*60 + "\n")
        
        skipped_subjects = []
        
        for subject_id in self.subjects:
            try:
                features, labels = self.extract_features_for_subject(subject_id)
                
                # Only add if we got valid data
                if len(features) > 0 and len(labels) > 0:
                    features_dict[subject_id] = features
                    labels_dict[subject_id] = labels
                else:
                    skipped_subjects.append(subject_id)
                    
            except Exception as e:
                print(f"S{subject_id}: ERROR - {e}")
                skipped_subjects.append(subject_id)
                continue
        
        print(f"\n✓ Successfully processed {len(features_dict)} subjects")
        if skipped_subjects:
            print(f"⚠ Skipped {len(skipped_subjects)} subjects: {skipped_subjects}")
        
        if len(features_dict) < 3:
            print("\n❌ ERROR: Not enough subjects with valid data!")
            print("   Need at least 3 subjects for LOSO cross-validation.")
            print("\nPossible issues:")
            print("  - Check that .pkl files contain wrist sensor data")
            print("  - Verify label data exists")
            print("  - Try loading one subject manually to debug")
            raise ValueError("Insufficient subjects with valid data")
        
        return features_dict, labels_dict
    
    def train_and_evaluate_model(self, 
                                model_name: str,
                                model,
                                features_dict: Dict[int, np.ndarray],
                                labels_dict: Dict[int, np.ndarray]) -> Dict:
        """
        Train and evaluate model using LOSO cross-validation
        
        Returns:
            results dictionary with metrics
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Check if we have enough subjects
        available_subjects = [s for s in self.subjects if s in features_dict]
        
        if len(available_subjects) < 3:
            print(f"❌ ERROR: Only {len(available_subjects)} subjects available")
            print("   Need at least 3 subjects for LOSO cross-validation")
            return {
                'model_name': model_name,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'error': 'Insufficient subjects'
            }
        
        print(f"Using {len(available_subjects)} subjects for LOSO CV")
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        subject_accuracies = []
        subject_f1s = []
        
        # LOSO Cross-validation
        for test_subject in available_subjects:
            
            # Prepare training data (all subjects except test subject)
            X_train_list = []
            y_train_list = []
            
            for train_subject in available_subjects:
                if train_subject != test_subject and train_subject in features_dict:
                    X_train_list.append(features_dict[train_subject])
                    y_train_list.append(labels_dict[train_subject])
            
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            
            # Test data
            X_test = features_dict[test_subject]
            y_test = labels_dict[test_subject]
            
            # Validate we have both classes in training data
            unique_train_classes = np.unique(y_train)
            unique_test_classes = np.unique(y_test)
            
            if len(unique_train_classes) < 2:
                print(f"  S{test_subject}: SKIP (training data has only {len(unique_train_classes)} class)")
                continue
            
            if len(unique_test_classes) < 1:
                print(f"  S{test_subject}: SKIP (test data is empty)")
                continue
            
            # Handle NaN/Inf values
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Get probabilities if available - with comprehensive error handling
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob_raw = model.predict_proba(X_test_scaled)
                    
                    # Handle different probability array shapes
                    if isinstance(y_prob_raw, np.ndarray):
                        if y_prob_raw.ndim == 1:
                            # 1-D array - use as is
                            y_prob = y_prob_raw
                        elif y_prob_raw.ndim == 2:
                            if y_prob_raw.shape[1] == 1:
                                # Only one class - use the single column
                                y_prob = y_prob_raw[:, 0]
                            elif y_prob_raw.shape[1] >= 2:
                                # Binary or multiclass - get positive class (index 1)
                                y_prob = y_prob_raw[:, 1]
                            else:
                                # Empty second dimension
                                y_prob = y_pred.astype(float)
                        else:
                            # Unexpected dimensionality
                            y_prob = y_pred.astype(float)
                    else:
                        # Not a numpy array
                        y_prob = y_pred.astype(float)
                        
                elif hasattr(model, 'decision_function'):
                    # For models like SVM with decision function
                    y_prob = model.decision_function(X_test_scaled)
                    if y_prob.ndim > 1:
                        y_prob = y_prob.flatten()
                else:
                    # No probability estimates available
                    y_prob = y_pred.astype(float)
                    
            except Exception as e:
                # If any probability extraction fails, use predictions
                print(f"    Warning: Could not extract probabilities ({e}), using predictions")
                y_prob = y_pred.astype(float)
            
            # Store results
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)
            all_probabilities.extend(y_prob)
            
            # Calculate subject-specific metrics
            subj_acc = accuracy_score(y_test, y_pred)
            subj_f1 = f1_score(y_test, y_pred, average='binary')
            subject_accuracies.append(subj_acc)
            subject_f1s.append(subj_f1)
            
            print(f"  S{test_subject}: Acc={subj_acc:.3f}, F1={subj_f1:.3f}")
        
        # Calculate overall metrics
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Check if we have any results
        if len(all_predictions) == 0 or len(all_true_labels) == 0:
            print(f"\n⚠️  No valid predictions collected")
            return {
                'model_name': model_name,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'subject_accuracies': [],
                'subject_f1s': [],
                'predictions': np.array([]),
                'true_labels': np.array([]),
                'probabilities': np.array([]),
                'error': 'No valid predictions'
            }
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(all_true_labels, all_predictions),
            'f1_score': f1_score(all_true_labels, all_predictions, average='binary', zero_division=0),
            'subject_accuracies': subject_accuracies,
            'subject_f1s': subject_f1s,
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'probabilities': all_probabilities,
            'classification_report': classification_report(all_true_labels, all_predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(all_true_labels, all_predictions)
        }
        
        # Calculate AUC if probabilities available
        try:
            if len(np.unique(all_probabilities)) > 1 and len(np.unique(all_true_labels)) > 1:
                results['auc'] = roc_auc_score(all_true_labels, all_probabilities)
        except Exception as e:
            # AUC calculation failed, skip it
            pass
        
        print(f"\n{'─'*60}")
        print(f"Overall Results:")
        if 'error' in results:
            print(f"  ERROR: {results['error']}")
        else:
            acc_std = np.std(subject_accuracies) if len(subject_accuracies) > 0 else 0.0
            f1_std = np.std(subject_f1s) if len(subject_f1s) > 0 else 0.0
            print(f"  Accuracy: {results['accuracy']:.4f} (±{acc_std:.4f})")
            print(f"  F1-Score: {results['f1_score']:.4f} (±{f1_std:.4f})")
            if 'auc' in results:
                print(f"  AUC: {results['auc']:.4f}")
        print(f"{'─'*60}")
        
        return results
    
    def run_all_models(self, features_dict: Dict, labels_dict: Dict) -> Dict:
        """
        Run all models and compare
        
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Models from paper
        print("\n" + "="*60)
        print("BASELINE MODELS (from paper)")
        print("="*60)
        
        models_paper = {
            'Decision Tree': DecisionTreeClassifier(
                criterion='entropy',
                min_samples_split=20,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                criterion='entropy',
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            'LDA': LinearDiscriminantAnalysis(),
            'kNN': KNeighborsClassifier(n_neighbors=9)
        }
        
        for name, model in models_paper.items():
            results[name] = self.train_and_evaluate_model(
                name, model, features_dict, labels_dict
            )
        
        # Enhanced models
        print("\n" + "="*60)
        print("ENHANCED MODELS")
        print("="*60)
        
        models_enhanced = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
        }
        
        for name, model in models_enhanced.items():
            results[name] = self.train_and_evaluate_model(
                name, model, features_dict, labels_dict
            )
        
        return results
    
    def plot_results(self, results: Dict, output_dir: Path = None):
        """
        Create visualizations of results
        """
        if output_dir is None:
            output_dir = Path("/mnt/user-data/outputs")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Model comparison barplot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        f1_scores = [results[m]['f1_score'] for m in models]
        
        # Accuracy
        axes[0].bar(models, accuracies, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0.7, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # F1-score
        axes[1].bar(models, f1_scores, color='coral', alpha=0.7)
        axes[1].set_ylabel('F1-Score', fontsize=12)
        axes[1].set_title('Model Comparison - F1-Score', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0.7, 1.0])
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(f1_scores):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrix for best model
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_results = results[best_model]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = best_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Non-Stress', 'Stress'],
                   yticklabels=['Non-Stress', 'Stress'])
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_matrix_{best_model.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC curve for models with probabilities
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for model_name, res in results.items():
            if 'auc' in res:
                fpr, tpr, _ = roc_curve(res['true_labels'], res['probabilities'])
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={res['auc']:.3f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Plots saved to {output_dir}")
    
    def save_results(self, results: Dict, features_dict: Dict, 
                    output_dir: Path = None):
        """
        Save all results to files
        """
        if output_dir is None:
            output_dir = Path("/mnt/user-data/outputs")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save summary
        summary_data = []
        for model_name, res in results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': res['accuracy'],
                'F1-Score': res['f1_score'],
                'AUC': res.get('auc', 'N/A'),
                'Accuracy_Std': np.std(res['subject_accuracies']),
                'F1_Std': np.std(res['subject_f1s'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        summary_df.to_csv(output_dir / 'results_summary.csv', index=False)
        
        # Save detailed results
        with open(output_dir / 'detailed_results.txt', 'w') as f:
            f.write("WESAD STRESS DETECTION - DETAILED RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for model_name, res in results.items():
                f.write(f"\n{model_name}\n")
                f.write("-"*60 + "\n")
                f.write(f"Accuracy: {res['accuracy']:.4f} (±{np.std(res['subject_accuracies']):.4f})\n")
                f.write(f"F1-Score: {res['f1_score']:.4f} (±{np.std(res['subject_f1s']):.4f})\n")
                if 'auc' in res:
                    f.write(f"AUC: {res['auc']:.4f}\n")
                f.write(f"\nClassification Report:\n")
                f.write(res['classification_report'])
                f.write("\n\n")
        
        # Save feature names for interpretability
        feature_names = get_feature_names(self.feature_extractor)
        with open(output_dir / 'feature_names.txt', 'w') as f:
            f.write("Feature Names (in order):\n")
            f.write("="*60 + "\n")
            for i, name in enumerate(feature_names, 1):
                f.write(f"{i}. {name}\n")
        
        print(f"\n✓ Results saved to {output_dir}")


def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("WESAD STRESS DETECTION - ENHANCED ML PIPELINE")
    print("="*60)
    
    # Try to load configuration
    try:
        import config
        data_dir = config.get_data_dir()
        print(f"\n✓ Using data directory: {data_dir}")
        
        # Check dataset
        if not config.check_dataset(data_dir):
            print("\n❌ Dataset check failed. Please fix the issues above.")
            return
    except ImportError:
        print("\n⚠️  config.py not found, using auto-detection")
        data_dir = None
    
    # Initialize pipeline
    pipeline = WESADPipeline(data_dir=data_dir)
    
    # Load and extract features for all subjects
    features_dict, labels_dict = pipeline.load_all_data()
    
    # Print dataset statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total_windows = sum(len(f) for f in features_dict.values())
    total_stress = sum(np.sum(l == 1) for l in labels_dict.values())
    total_nonstress = sum(np.sum(l == 0) for l in labels_dict.values())
    
    print(f"Total subjects: {len(features_dict)}")
    print(f"Total windows: {total_windows}")
    print(f"Stress windows: {total_stress} ({total_stress/total_windows*100:.1f}%)")
    print(f"Non-stress windows: {total_nonstress} ({total_nonstress/total_windows*100:.1f}%)")
    print(f"Feature dimensionality: {features_dict[list(features_dict.keys())[0]].shape[1]}")
    
    # Run all models
    results = pipeline.run_all_models(features_dict, labels_dict)
    
    # Create visualizations
    pipeline.plot_results(results)
    
    # Save results
    pipeline.save_results(results, features_dict)
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60 + "\n")
    
    comparison_data = []
    for model_name, res in results.items():
        if 'error' not in res:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{res['accuracy']:.4f}",
                'F1-Score': f"{res['f1_score']:.4f}",
                'AUC': f"{res.get('auc', 'N/A')}"
            })
        else:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': 'ERROR',
                'F1-Score': 'ERROR',
                'AUC': 'N/A'
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Comparison to paper (only for models without errors)
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if len(valid_results) > 0:
        print("\n" + "="*60)
        print("COMPARISON TO PAPER (Wrist-based, Binary Classification)")
        print("="*60)
        print("Paper results: ~88% accuracy with wrist physiological data")
        
        best_model = max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'])
        best_acc = valid_results[best_model]['accuracy']
        
        print(f"Our best result: {best_acc:.4f} accuracy with {best_model}")
        
        if best_acc >= 0.88:
            print("✓ MATCHED OR EXCEEDED PAPER BASELINE!")
        else:
            print(f"Gap to paper: {0.88 - best_acc:.4f}")
            print("Suggestions: Try hyperparameter tuning, feature selection, or deep learning")
    else:
        print("\n⚠️  No models completed successfully")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
