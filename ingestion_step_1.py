import os
import mne

# Path to the downloaded PhysioNet EDF file
EDF_PATH = r"C:\Users\Ayushi Ghosh\OneDrive\Desktop\EEG\edf files\S001R01.edf"

def load_eeg_file(path):
    # Verify file existence before reading
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target EDF file not found at: {path}")
    
    print(f"Loading EEG data from: {path}")
    
    # Read raw recording into memory
    raw_data = mne.io.read_raw_edf(path, preload=True)
    
    # Apply standard electrode scalp locations
    montage = mne.channels.make_standard_montage('standard_1005')
    raw_data.set_montage(montage, match_case=False, on_missing='ignore')
    
    return raw_data


if __name__ == "__main__":
    # Ingest data
    eeg = load_eeg_file(EDF_PATH)
    
    # Extract metadata
    s_freq = eeg.info['sfreq']
    ch_names = eeg.ch_names
    duration_sec = eeg.n_times / s_freq
    
    # Print Summary
    print("\n" + "=" * 50)
    print("           EEG DATASET INGESTION SUMMARY")
    print("=" * 50)
    print(f"Sampling Frequency : {s_freq} Hz")
    print(f"Total Duration     : {duration_sec:.2f} seconds")
    print(f"Total Timepoints   : {eeg.n_times}")
    print(f"Channel Count      : {len(ch_names)}")
    print("-" * 50)
    print("Channel Layout (64 Electrodes):")
    
    # Print all 64 channels neatly formatted in 8 columns
    for i in range(0, len(ch_names), 8):
        line = "  ".join(f"{ch:<6}" for ch in ch_names[i:i+8])
        print(f"  {line}")
        
    print("=" * 50 + "\n")