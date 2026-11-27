import warnings
import numpy as np
import matplotlib.pyplot as plt
import mne

warnings.filterwarnings('ignore')

from ingestion_step_1 import load_eeg_file, EDF_PATH
from step_2_signal_processing import preprocess_eeg_data


def run_spectral_analysis(clean_eeg):
    """
    Step 3: Computes Power Spectral Density (PSD) using Welch's method,
    extracts relative band power (Delta, Theta, Alpha, Beta, Gamma),
    and renders 2D scalp activation topomaps.
    """
    print("[+] Starting Step 3: Spectral Power & 2D Scalp Topography...")

    # FIX: Clean PhysioNet channel names (strip trailing dots like 'Fc5.' -> 'FC5')
    # and attach standard 10-05 montage so 2D topomaps have valid coordinates
    mne.datasets.eegbci.standardize(clean_eeg)
    montage = mne.channels.make_standard_montage('standard_1005')
    clean_eeg.set_montage(montage, match_case=False, on_missing='ignore')

    # Define standard physiological brainwave frequency bands
    freq_bands = {
        'Delta (0.5-4 Hz)': (0.5, 4.0),
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-12 Hz)': (8.0, 12.0),
        'Beta (12-30 Hz)': (12.0, 30.0),
        'Gamma (30-45 Hz)': (30.0, 45.0)
    }

    # 1. Compute Power Spectral Density (PSD) via Welch's method
    print("  -> Computing Power Spectral Density (Welch's method)...")
    psd = clean_eeg.compute_psd(method='welch', fmin=0.5, fmax=45.0, verbose=False)
    
    psds, freqs = psd.get_data(return_freqs=True)
    total_power = np.sum(psds)

    # 2. Calculate and print relative band power percentages
    print("\n--- FREQUENCY BAND POWER BREAKDOWN ---")
    for band_name, (fmin, fmax) in freq_bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.sum(psds[:, idx_band])
        rel_power = (band_power / total_power) * 100
        print(f"  {band_name:<18} : {rel_power:.2f}%")
    print("--------------------------------------\n")

    # 3. Plot PSD Frequency Spectrum Curve
    print("  -> Rendering PSD spectrum graph...")
    fig_psd = psd.plot(show=False)
    fig_psd.suptitle("Power Spectral Density (0.5 - 45.0 Hz)", fontsize=13)

    # 4. Plot 2D Scalp Activation Topomaps
    print("  -> Generating 2D Scalp Topomaps...")
    fig_topo = psd.plot_topomap(bands=freq_bands, ch_type='eeg', show=False)
    fig_topo.suptitle("2D Scalp Activation Topomaps Across Brainwave Bands", fontsize=13)

    print("[+] Step 3 Complete. Displaying visual plots...\n")
    plt.show()

    return psd, fig_psd, fig_topo


if __name__ == "__main__":
    raw_data = load_eeg_file(EDF_PATH)
    cleaned_eeg = preprocess_eeg_data(raw_data)
    run_spectral_analysis(cleaned_eeg)