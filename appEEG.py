import os
import tempfile
import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EEG Signal Processing & Topomap Dashboard",
    layout="wide"
)

st.title("EEG Signal Processing & Topographic Visualization Dashboard")
st.markdown("Upload a multi-channel `.edf` recording to clean artifacts, inspect frequency spectra, and view 2D scalp topomaps.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Upload EEG Recording")
uploaded_file = st.sidebar.file_uploader("Choose an .edf file", type=["edf"])

st.sidebar.header("2. Signal Filtering Controls")
low_cut = st.sidebar.slider("High-Pass Cutoff (Hz)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
high_cut = st.sidebar.slider("Low-Pass Cutoff (Hz)", min_value=20.0, max_value=70.0, value=45.0, step=1.0)
apply_notch = st.sidebar.checkbox("Apply 50 Hz Power Grid Notch Filter", value=True)

# --- PROCESSING PIPELINE FUNCTION ---
@st.cache_data
def process_eeg_file(file_bytes, l_freq, h_freq, use_notch):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(file_bytes)
        temp_path = tmp_file.name

    try:
        # Step 1: Data Ingestion & Montage Mapping
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, match_case=False, on_missing='ignore')

        # Step 2: Signal Preprocessing & Artifact Cleaning
        raw_clean = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
        if use_notch:
            raw_clean.notch_filter(freqs=50.0, verbose=False)

        # Step 3: Spectral Power (Welch's PSD)
        psd = raw_clean.compute_psd(method='welch', fmin=l_freq, fmax=h_freq, verbose=False)

        return raw, raw_clean, psd
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- MAIN DASHBOARD DISPLAY ---
if uploaded_file is not None:
    with st.spinner("Processing brainwave signals..."):
        raw_orig, raw_clean, psd = process_eeg_file(
            uploaded_file.getvalue(), low_cut, high_cut, apply_notch
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sampling Rate", f"{raw_clean.info['sfreq']} Hz")
    col2.metric("Active Channels", len(raw_clean.ch_names))
    col3.metric("Duration", f"{raw_clean.n_times / raw_clean.info['sfreq']:.1f} sec")
    col4.metric("Bandpass Range", f"{low_cut} - {high_cut} Hz")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "1. Cleaned Waveforms",
        "2. Power Spectral Density (PSD)",
        "3. 2D Scalp Topomaps"
    ])

    # --- TAB 1: WAVEFORMS ---
    with tab1:
        st.subheader("Raw vs. Cleaned Time-Series Waveforms")
        selected_ch = st.selectbox("Select Electrode Channel to Inspect:", raw_clean.ch_names, index=0)
        time_window = st.slider("Select Time Window (Seconds):", 0, int(raw_clean.times[-1]) - 5, (0, 5))

        start_idx = int(time_window[0] * raw_clean.info['sfreq'])
        stop_idx = int(time_window[1] * raw_clean.info['sfreq'])

        times = raw_clean.times[start_idx:stop_idx]
        orig_data = raw_orig.get_data(picks=[selected_ch])[0, start_idx:stop_idx] * 1e6
        clean_data = raw_clean.get_data(picks=[selected_ch])[0, start_idx:stop_idx] * 1e6

        fig_wave, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax1.plot(times, orig_data, color="gray", linewidth=0.9)
        ax1.set_title(f"Raw Signal - Channel {selected_ch}")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(times, clean_data, color="#1f77b4", linewidth=1.1)
        ax2.set_title(f"Filtered Signal ({low_cut}–{high_cut} Hz) - Channel {selected_ch}")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Amplitude (µV)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_wave)

    # --- TAB 2: PSD & FREQUENCY BANDS ---
    with tab2:
        st.subheader("Power Spectral Density (Welch's Method)")
        fig_psd = psd.plot(show=False)
        st.pyplot(fig_psd)

        st.markdown("#### Physiological Brainwave Frequency Bands")
        bands = {
            "Delta (0.5–4 Hz)": (0.5, 4.0),
            "Theta (4–8 Hz)": (4.0, 8.0),
            "Alpha (8–12 Hz)": (8.0, 12.0),
            "Beta (12–30 Hz)": (12.0, 30.0),
            "Gamma (30–45 Hz)": (30.0, 45.0)
        }

        psds, freqs = psd.get_data(return_freqs=True)
        band_powers = []
        for band_name, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            mean_power = np.mean(psds[:, idx]) * 1e12
            band_powers.append(mean_power)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        ax_bar.bar(bands.keys(), band_powers, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"])
        ax_bar.set_ylabel("Mean Relative Power (µV²/Hz)")
        ax_bar.set_title("Average Power Distribution Across Brainwave Bands")
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_bar)

    # --- TAB 3: 2D SCALP TOPOMAPS ---
    with tab3:
        st.subheader("2D Spatial Scalp Activation Topomaps")
        st.markdown("Color intensity highlights cortical activity across standard 10-10 electrode positions.")

        topo_bands = {
            'Delta (1–4 Hz)': (1, 4),
            'Theta (4–8 Hz)': (4, 8),
            'Alpha (8–12 Hz)': (8, 12),
            'Beta (12–30 Hz)': (12, 30),
            'Gamma (30–45 Hz)': (30, 45)
        }

        valid_bands = {k: v for k, v in topo_bands.items() if v[0] >= low_cut and v[1] <= high_cut}

        if valid_bands:
            fig_topo = psd.plot_topomap(bands=valid_bands, ch_type='eeg', normalize=True, show=False)
            st.pyplot(fig_topo)
        else:
            st.warning("Adjust your sidebar frequency sliders (e.g., 1.0 to 45.0 Hz) to display standard brainwave topomaps.")

else:
    st.info("Please upload an `.edf` file from the left sidebar to launch the analysis pipeline.")