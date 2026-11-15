import warnings
import numpy as np
import mne

warnings.filterwarnings('ignore')

from ingestion_step_1 import load_eeg_file, EDF_PATH

def auto_detect_bad_channels(raw_inst, z_threshold=3.5):
    
    data = raw_inst.get_data()
    ch_stds = np.std(data, axis=1)
    
    # Check for NaN or Inf in std calculations
    if np.any(np.isnan(ch_stds)) or np.any(np.isinf(ch_stds)):
        return []

    # Flag flat channels (< 1e-12)
    flat_mask = ch_stds < 1e-12
    valid_stds = ch_stds[~flat_mask]
    
    if len(valid_stds) == 0:
        return []
        
    mean_std = np.mean(valid_stds)
    std_std = np.std(valid_stds)
    
    if std_std == 0:
        return []

    # Calculate safe Z-scores
    z_scores = np.abs((ch_stds - mean_std) / std_std)
    bad_mask = (z_scores > z_threshold) | flat_mask
    
    bad_channels = [raw_inst.ch_names[i] for i, is_bad in enumerate(bad_mask) if is_bad]
    return bad_channels


def preprocess_eeg_data(raw_recording):
  
    print("[+] Starting Step 2: Signal Preprocessing...")
    clean_data = raw_recording.copy()

    # 1. Bandpass Filter (1.0 - 45.0 Hz)
    print("  -> Applying 1.0 - 45.0 Hz Bandpass Filter...")
    clean_data.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)

    # 2. Notch Filter (50.0 Hz Power Line Noise)
    print("  -> Applying 50.0 Hz Notch Filter...")
    clean_data.notch_filter(freqs=50.0, verbose=False)

    # 3. Bad Channel Detection & Spline Reconstruction
    bad_sensors = auto_detect_bad_channels(clean_data, z_threshold=3.5)
    
    if bad_sensors:
        clean_data.info['bads'] = bad_sensors
        print(f"-Detected bad channel(s): {bad_sensors}")
        print("-Reconstructing via 3D Spherical Splines")
        try:
            clean_data.interpolate_bads(origin=(0.0, 0.0, 0.0), verbose=False)
        except Exception as e:
            print(f"(!) Skipping interpolation due to sensor geometry limit: {e}")
            clean_data.info['bads'] = []
    else:
        print("\u2705 Signals clean. No bad channels detected beyond threshold.")

    print("✓ Step 2 Completed\n")
    return clean_data

if __name__ == "__main__":
    raw_data = load_eeg_file(EDF_PATH)
    cleaned_eeg = preprocess_eeg_data(raw_data)