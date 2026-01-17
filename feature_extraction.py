"""
Enhanced Feature Extraction for WESAD Wrist Data
================================================

Features from Paper + Novel Enhancements:
1. Paper's features (Table 1)
2. Time-frequency features (wavelets)
3. Nonlinear features
4. Cross-modal features
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import pywt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class WristFeatureExtractor:
    """
    Extract features from wrist-worn device sensors
    Based on WESAD paper + enhancements
    """
    
    def __init__(self, 
                 window_size_acc=5,      # 5 seconds for ACC (paper)
                 window_size_physio=60,   # 60 seconds for physio (paper)
                 window_shift=0.25):      # 0.25 second shift
        
        self.window_size_acc = window_size_acc
        self.window_size_physio = window_size_physio
        self.window_shift = window_shift
        
        # Sampling rates from Empatica E4
        self.fs_bvp = 64
        self.fs_eda = 4
        self.fs_temp = 4
        self.fs_acc = 32
        
    def extract_all_features(self, wrist_data: Dict, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract all features from wrist data
        
        Args:
            wrist_data: Dict with keys 'ACC', 'BVP', 'EDA', 'TEMP'
            labels: Label array aligned with sensor data
            
        Returns:
            features: (n_windows, n_features) array
            window_labels: (n_windows,) array
        """
        
        # Get all sensors and ensure correct shapes
        acc = wrist_data['ACC']  # Shape: (n_samples, 3) or (n_samples,)
        bvp = wrist_data['BVP']  # Shape: (n_samples,) 
        eda = wrist_data['EDA']  # Shape: (n_samples,)
        temp = wrist_data['TEMP']  # Shape: (n_samples,)
        
        # Ensure correct shapes
        if bvp.ndim > 1:
            bvp = bvp.flatten()
        if eda.ndim > 1:
            eda = eda.flatten()
        if temp.ndim > 1:
            temp = temp.flatten()
        if acc.ndim == 1:
            # If ACC is 1-D, reshape to (n_samples, 1) and pad with zeros
            acc = np.column_stack([acc, np.zeros_like(acc), np.zeros_like(acc)])
        elif acc.ndim > 2:
            acc = acc.reshape(-1, 3)
        
        # Use physiological window size (60s) as base
        window_samples = int(self.window_size_physio * self.fs_eda)  # Use lowest fs
        shift_samples = int(self.window_shift * self.fs_eda)
        
        features_list = []
        labels_list = []
        
        # Sliding window
        n_windows = (len(eda) - window_samples) // shift_samples + 1
        
        for i in range(n_windows):
            start_idx = i * shift_samples
            end_idx = start_idx + window_samples
            
            # Get window label (majority vote)
            window_labels = labels[start_idx:end_idx]
            
            # Labels are original WESAD labels:
            # 0 = transient (filter out)
            # 1 = baseline (keep as non-stress)
            # 2 = stress (keep as stress)
            # 3 = amusement (keep as non-stress)
            # 4 = meditation (filter out)
            
            # Filter out transient (0) and meditation (4)
            valid_labels = window_labels[(window_labels > 0) & (window_labels < 4)]
            
            # Skip if not enough valid labels (>50% must be valid)
            if len(valid_labels) < len(window_labels) * 0.5:
                continue
            
            # Get most common label (will be 1, 2, or 3)
            if len(valid_labels) == 0:
                continue
            
            # Get most frequent label
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            label = unique_labels[np.argmax(counts)]  # Returns actual label value (1, 2, or 3)
            
            # Extract indices for each sensor (different sampling rates)
            # EDA indices (4 Hz - base)
            eda_start, eda_end = start_idx, end_idx
            
            # BVP indices (64 Hz)
            bvp_start = int(start_idx * self.fs_bvp / self.fs_eda)
            bvp_end = int(end_idx * self.fs_bvp / self.fs_eda)
            
            # TEMP indices (4 Hz)
            temp_start, temp_end = eda_start, eda_end
            
            # ACC indices (32 Hz) - use 5s window as per paper
            acc_window_samples = int(self.window_size_acc * self.fs_acc)
            acc_start = int(start_idx * self.fs_acc / self.fs_eda)
            acc_end = acc_start + acc_window_samples
            
            # Ensure indices are valid
            if (bvp_end > len(bvp) or acc_end > len(acc) or 
                eda_end > len(eda) or temp_end > len(temp)):
                continue
            
            # Extract window data
            bvp_window = bvp[bvp_start:bvp_end]
            eda_window = eda[eda_start:eda_end]
            temp_window = temp[temp_start:temp_end]
            acc_window = acc[acc_start:acc_end]
            
            # Extract features for each modality
            feature_vector = []
            
            # ACC features
            acc_feats = self.extract_acc_features(acc_window)
            feature_vector.extend(acc_feats)
            
            # BVP features (heart rate related)
            bvp_feats = self.extract_bvp_features(bvp_window)
            feature_vector.extend(bvp_feats)
            
            # EDA features
            eda_feats = self.extract_eda_features(eda_window)
            feature_vector.extend(eda_feats)
            
            # TEMP features
            temp_feats = self.extract_temp_features(temp_window)
            feature_vector.extend(temp_feats)
            
            # ENHANCED: Cross-modal features
            cross_feats = self.extract_cross_modal_features(
                bvp_window, eda_window, temp_window
            )
            feature_vector.extend(cross_feats)
            
            features_list.append(feature_vector)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    # ==================== ACC Features ====================
    
    def extract_acc_features(self, acc: np.ndarray) -> List[float]:
        """
        Extract acceleration features (Paper: Table 1)
        
        Args:
            acc: (n_samples, 3) acceleration data
            
        Returns:
            List of features
        """
        features = []
        
        # Ensure correct shape
        if acc.ndim == 1:
            # Convert 1-D to 3-D (pad with zeros)
            acc = np.column_stack([acc, np.zeros_like(acc), np.zeros_like(acc)])
        elif acc.shape[1] != 3:
            # If not 3 columns, pad or truncate
            if acc.shape[1] < 3:
                pad_width = ((0, 0), (0, 3 - acc.shape[1]))
                acc = np.pad(acc, pad_width, mode='constant')
            else:
                acc = acc[:, :3]
        
        # Per-axis features
        for axis in range(3):
            axis_data = acc[:, axis]
            features.append(np.mean(axis_data))              # μ_ACC,i
            features.append(np.std(axis_data))               # σ_ACC,i
            features.append(np.sum(np.abs(axis_data)))       # ||∫ACC,i||
            
            # Peak frequency
            if len(axis_data) > 10:
                try:
                    freqs, psd = scipy_signal.periodogram(axis_data, fs=self.fs_acc)
                    peak_freq = freqs[np.argmax(psd)]
                    features.append(peak_freq)                   # f^peak_ACC,i
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # 3D magnitude features
        acc_3d = np.sqrt(np.sum(acc**2, axis=1))
        features.append(np.mean(acc_3d))                     # μ_ACC,3D
        features.append(np.std(acc_3d))                      # σ_ACC,3D
        features.append(np.sum(np.abs(acc_3d)))              # ||∫ACC,3D||
        
        # ENHANCED: Additional statistical features
        features.append(skew(acc_3d))                        # Skewness
        features.append(kurtosis(acc_3d))                    # Kurtosis
        features.append(np.percentile(acc_3d, 25))           # 25th percentile
        features.append(np.percentile(acc_3d, 75))           # 75th percentile
        
        return features
    
    # ==================== BVP Features ====================
    
    def extract_bvp_features(self, bvp: np.ndarray) -> List[float]:
        """
        Extract BVP/heart rate features (Paper: Table 1, ECG section)
        
        Args:
            bvp: (n_samples,) BVP signal
            
        Returns:
            List of features
        """
        features = []
        
        # Detect peaks (heart beats)
        peaks = self.detect_bvp_peaks(bvp)
        
        if len(peaks) < 2:
            # Not enough peaks - return zeros
            return [0.0] * 20
        
        # Heart rate
        rr_intervals = np.diff(peaks) / self.fs_bvp  # RR intervals in seconds
        hr = 60.0 / rr_intervals  # Heart rate in BPM
        
        features.append(np.mean(hr))                         # μ_HR
        features.append(np.std(hr))                          # σ_HR
        
        # HRV features
        hrv = rr_intervals * 1000  # Convert to ms
        features.append(np.mean(hrv))                        # μ_HRV
        features.append(np.std(hrv))                         # σ_HRV
        
        # NN50, pNN50
        nn50 = np.sum(np.abs(np.diff(hrv)) > 50)
        pnn50 = nn50 / len(hrv) * 100 if len(hrv) > 0 else 0
        features.append(nn50)                                # NN50
        features.append(pnn50)                               # pNN50
        
        # RMSSD
        rmssd = np.sqrt(np.mean(np.diff(hrv)**2))
        features.append(rmssd)                               # rms_HRV
        
        # Frequency domain features (if enough data)
        if len(hrv) > 10:
            # Interpolate for spectral analysis
            time_hrv = np.cumsum(rr_intervals)
            time_uniform = np.arange(0, time_hrv[-1], 0.25)  # 4 Hz
            hrv_interp = np.interp(time_uniform, time_hrv, hrv)
            
            # Compute PSD
            freqs, psd = scipy_signal.periodogram(hrv_interp, fs=4, window='hamming')
            
            # Frequency bands
            ulf = self._band_power(freqs, psd, 0.01, 0.04)   # f^ULF_HRV
            lf = self._band_power(freqs, psd, 0.04, 0.15)    # f^LF_HRV
            hf = self._band_power(freqs, psd, 0.15, 0.4)     # f^HF_HRV
            uhf = self._band_power(freqs, psd, 0.4, 1.0)     # f^UHF_HRV
            
            features.append(ulf)
            features.append(lf)
            features.append(hf)
            features.append(uhf)
            
            # LF/HF ratio
            lf_hf_ratio = lf / hf if hf > 0 else 0
            features.append(lf_hf_ratio)                     # f^LF/HF_HRV
            
            # Normalized LF and HF
            total_power = lf + hf
            lf_norm = lf / total_power if total_power > 0 else 0
            hf_norm = hf / total_power if total_power > 0 else 0
            features.append(lf_norm)                         # LF_norm
            features.append(hf_norm)                         # HF_norm
        else:
            features.extend([0.0] * 8)
        
        # ENHANCED: Poincaré plot features
        sd1, sd2 = self._poincare_features(hrv)
        features.append(sd1)
        features.append(sd2)
        features.append(sd1 / sd2 if sd2 > 0 else 0)        # SD1/SD2 ratio
        
        return features
    
    def detect_bvp_peaks(self, bvp: np.ndarray) -> np.ndarray:
        """Detect peaks in BVP signal (heart beats)"""
        # Ensure BVP is 1-D
        if bvp.ndim > 1:
            bvp = bvp.flatten()
        
        if len(bvp) < 10:
            return np.array([])
        
        # Simple peak detection
        # In practice, might want to use more sophisticated method
        distance = int(0.6 * self.fs_bvp)  # Minimum 100 BPM
        
        try:
            peaks, _ = scipy_signal.find_peaks(bvp, distance=distance, prominence=np.std(bvp)*0.5)
            return peaks
        except Exception as e:
            # If peak detection fails, return empty array
            return np.array([])
    
    def _band_power(self, freqs: np.ndarray, psd: np.ndarray, 
                    low: float, high: float) -> float:
        """Calculate power in frequency band"""
        idx = np.logical_and(freqs >= low, freqs < high)
        return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0
    
    def _poincare_features(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """Calculate Poincaré plot features (SD1, SD2)"""
        if len(rr_intervals) < 2:
            return 0.0, 0.0
        
        # SD1: standard deviation perpendicular to line of identity
        diff_rr = np.diff(rr_intervals)
        sd1 = np.sqrt(np.var(diff_rr) / 2)
        
        # SD2: standard deviation along line of identity
        sd2 = np.sqrt(2 * np.var(rr_intervals) - np.var(diff_rr) / 2)
        
        return sd1, sd2
    
    # ==================== EDA Features ====================
    
    def extract_eda_features(self, eda: np.ndarray) -> List[float]:
        """
        Extract EDA features (Paper: Table 1)
        
        Args:
            eda: (n_samples,) EDA signal
            
        Returns:
            List of features
        """
        features = []
        
        # Basic statistical features
        features.append(np.mean(eda))                        # μ_EDA
        features.append(np.std(eda))                         # σ_EDA
        features.append(np.min(eda))                         # min_EDA
        features.append(np.max(eda))                         # max_EDA
        features.append(np.max(eda) - np.min(eda))           # range_EDA
        
        # Slope (linear regression)
        if len(eda) > 1:
            x = np.arange(len(eda))
            slope = np.polyfit(x, eda, 1)[0]
            features.append(slope)                           # ∂_EDA
        else:
            features.append(0.0)
        
        # Decompose into SCL (tonic) and SCR (phasic)
        scl, scr = self.decompose_eda(eda)
        
        # SCL features
        features.append(np.mean(scl))                        # μ_SCL
        features.append(np.std(scl))                         # σ_SCL
        
        # Correlation between SCL and time
        if len(scl) > 1:
            time = np.arange(len(scl))
            corr = np.corrcoef(scl, time)[0, 1]
            features.append(corr)                            # corr(SCL, t)
        else:
            features.append(0.0)
        
        # SCR features
        features.append(np.std(scr))                         # σ_SCR
        
        # Detect SCR peaks
        scr_peaks = self.detect_scr_peaks(scr)
        features.append(len(scr_peaks))                      # #SCR
        
        if len(scr_peaks) > 0:
            peak_amplitudes = scr[scr_peaks]
            features.append(np.sum(peak_amplitudes))         # Σ A^mp_SCR
            features.append(np.sum(scr_peaks) / self.fs_eda) # Σ t_SCR
            features.append(np.trapz(scr))                   # ∫_scr (area)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # ENHANCED: Wavelet features for EDA
        wavelet_feats = self._wavelet_features(eda, 'db4', self.fs_eda)
        features.extend(wavelet_feats)
        
        return features
    
    def decompose_eda(self, eda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose EDA into tonic (SCL) and phasic (SCR) components
        Using simple low-pass filtering approach
        """
        # Low-pass filter for SCL (< 0.05 Hz)
        b, a = scipy_signal.butter(3, 0.05 / (self.fs_eda / 2), btype='low')
        scl = scipy_signal.filtfilt(b, a, eda)
        
        # SCR is the residual
        scr = eda - scl
        
        return scl, scr
    
    def detect_scr_peaks(self, scr: np.ndarray) -> np.ndarray:
        """Detect peaks in SCR (skin conductance response)"""
        if len(scr) < 2:
            return np.array([])
        
        # Peaks should be positive and above threshold
        threshold = np.mean(scr) + 0.5 * np.std(scr)
        peaks, _ = scipy_signal.find_peaks(scr, height=threshold, distance=int(self.fs_eda))
        return peaks
    
    def _wavelet_features(self, signal_data: np.ndarray, 
                          wavelet: str, fs: float) -> List[float]:
        """
        Extract wavelet-based features (ENHANCED)
        
        Returns energy in different frequency bands
        """
        if len(signal_data) < 4:
            return [0.0] * 4
        
        try:
            # Discrete wavelet transform
            coeffs = pywt.wavedec(signal_data, wavelet, level=3)
            
            # Energy in each level
            energies = [np.sum(c**2) for c in coeffs]
            
            # Normalize
            total_energy = sum(energies)
            if total_energy > 0:
                energies = [e / total_energy for e in energies]
            
            return energies
        except:
            return [0.0] * 4
    
    # ==================== TEMP Features ====================
    
    def extract_temp_features(self, temp: np.ndarray) -> List[float]:
        """
        Extract temperature features (Paper: Table 1)
        
        Args:
            temp: (n_samples,) temperature signal
            
        Returns:
            List of features
        """
        features = []
        
        # Basic statistical features
        features.append(np.mean(temp))                       # μ_TEMP
        features.append(np.std(temp))                        # σ_TEMP
        features.append(np.min(temp))                        # min_TEMP
        features.append(np.max(temp))                        # max_TEMP
        features.append(np.max(temp) - np.min(temp))         # range_TEMP
        
        # Slope
        if len(temp) > 1:
            x = np.arange(len(temp))
            slope = np.polyfit(x, temp, 1)[0]
            features.append(slope)                           # ∂_TEMP
        else:
            features.append(0.0)
        
        # ENHANCED: Rate of change features
        if len(temp) > 1:
            temp_diff = np.diff(temp)
            features.append(np.mean(np.abs(temp_diff)))      # Mean absolute change
            features.append(np.max(np.abs(temp_diff)))       # Max absolute change
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    # ==================== Cross-Modal Features (ENHANCED) ====================
    
    def extract_cross_modal_features(self, bvp: np.ndarray, 
                                    eda: np.ndarray, 
                                    temp: np.ndarray) -> List[float]:
        """
        Extract features that capture relationships between modalities
        This is ENHANCED beyond the paper
        """
        features = []
        
        # Downsample BVP to match EDA frequency
        bvp_downsampled = scipy_signal.resample(bvp, len(eda))
        
        # Correlation between BVP and EDA
        if len(bvp_downsampled) == len(eda) and len(eda) > 1:
            corr_bvp_eda = np.corrcoef(bvp_downsampled, eda)[0, 1]
            features.append(corr_bvp_eda if not np.isnan(corr_bvp_eda) else 0.0)
        else:
            features.append(0.0)
        
        # Correlation between EDA and TEMP
        if len(eda) == len(temp) and len(eda) > 1:
            corr_eda_temp = np.corrcoef(eda, temp)[0, 1]
            features.append(corr_eda_temp if not np.isnan(corr_eda_temp) else 0.0)
        else:
            features.append(0.0)
        
        # Synchrony: Do BVP and EDA peak together?
        # Detect peaks in both
        bvp_peaks = self.detect_bvp_peaks(bvp_downsampled)
        _, scr = self.decompose_eda(eda)
        eda_peaks = self.detect_scr_peaks(scr)
        
        # Count peaks within 2-second windows
        if len(bvp_peaks) > 0 and len(eda_peaks) > 0:
            window_size = 2 * self.fs_eda  # 2 seconds
            synchronized = 0
            for bp in bvp_peaks:
                if np.any(np.abs(eda_peaks - bp) < window_size):
                    synchronized += 1
            sync_ratio = synchronized / len(bvp_peaks)
            features.append(sync_ratio)
        else:
            features.append(0.0)
        
        # Arousal index: combination of HR increase and EDA increase
        # (higher values = higher arousal)
        hr_normalized = (np.mean(bvp_downsampled) - np.min(bvp_downsampled)) / \
                       (np.max(bvp_downsampled) - np.min(bvp_downsampled) + 1e-6)
        eda_normalized = (np.mean(eda) - np.min(eda)) / \
                        (np.max(eda) - np.min(eda) + 1e-6)
        arousal_index = (hr_normalized + eda_normalized) / 2
        features.append(arousal_index)
        
        return features


def get_feature_names(extractor: WristFeatureExtractor) -> List[str]:
    """
    Get names for all features for interpretability
    """
    names = []
    
    # ACC features (3 axes + 3D)
    for axis in ['x', 'y', 'z']:
        names.extend([f'acc_{axis}_mean', f'acc_{axis}_std', 
                     f'acc_{axis}_absint', f'acc_{axis}_peak_freq'])
    names.extend(['acc_3d_mean', 'acc_3d_std', 'acc_3d_absint',
                 'acc_3d_skew', 'acc_3d_kurt', 'acc_3d_p25', 'acc_3d_p75'])
    
    # BVP features (20 features)
    names.extend(['hr_mean', 'hr_std', 'hrv_mean', 'hrv_std',
                 'nn50', 'pnn50', 'rmssd', 'ulf', 'lf', 'hf', 'uhf',
                 'lf_hf_ratio', 'lf_norm', 'hf_norm',
                 'sd1', 'sd2', 'sd1_sd2_ratio'])
    
    # EDA features
    names.extend(['eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range',
                 'eda_slope', 'scl_mean', 'scl_std', 'scl_time_corr',
                 'scr_std', 'scr_count', 'scr_amp_sum', 'scr_time_sum', 'scr_area'])
    names.extend([f'eda_wavelet_{i}' for i in range(4)])
    
    # TEMP features
    names.extend(['temp_mean', 'temp_std', 'temp_min', 'temp_max', 
                 'temp_range', 'temp_slope', 'temp_mean_change', 'temp_max_change'])
    
    # Cross-modal features
    names.extend(['corr_bvp_eda', 'corr_eda_temp', 'peak_sync_ratio', 'arousal_index'])
    
    return names


if __name__ == "__main__":
    print("Enhanced Feature Extraction Module Loaded")
    print(f"Total features to be extracted: ~{len(get_feature_names(WristFeatureExtractor()))}")
