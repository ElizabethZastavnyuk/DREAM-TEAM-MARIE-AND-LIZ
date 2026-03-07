# src/process_data_labeled.py

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import trapezoid

# ----------------------------
# CONFIG
# ----------------------------
FS_EEG = 256
FS_ACC = 50
FS_GYRO = 50

def bandpass_filter(x, fs, low=1.0, high=12.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def line_length(x):
    return float(np.sum(np.abs(np.diff(x))))

def bandpower(x, fs, fmin, fmax):
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), 256))
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return trapezoid(psd[mask], freqs[mask])

# ----------------------------
# LOAD TRIAL CSV
# ----------------------------
def load_trial(trial_folder):
    eeg_file = [f for f in os.listdir(trial_folder) if "_eeg.csv" in f][0]
    acc_file = [f for f in os.listdir(trial_folder) if "_acc.csv" in f][0]
    gyro_file = [f for f in os.listdir(trial_folder) if "_gyro.csv" in f][0]
    markers_file = [f for f in os.listdir(trial_folder) if "_markers.csv" in f][0]

    eeg = pd.read_csv(os.path.join(trial_folder, eeg_file)).to_numpy()
    acc = pd.read_csv(os.path.join(trial_folder, acc_file)).to_numpy()
    gyro = pd.read_csv(os.path.join(trial_folder, gyro_file)).to_numpy()
    markers = pd.read_csv(os.path.join(trial_folder, markers_file))

    return eeg, acc, gyro, markers

# ----------------------------
# EXTRACT FEATURES PER PERIOD
# ----------------------------
def extract_features(eeg, acc, gyro, markers):
    periods = []
    starts = markers[markers.marker == "period_start"].timestamp.values
    ends = markers[markers.marker == "period_end"].timestamp.values

    if len(starts) != len(ends):
        min_len = min(len(starts), len(ends))
        starts, ends = starts[:min_len], ends[:min_len]

    for i, (start, end) in enumerate(zip(starts, ends)):
        eeg_idx = np.where((np.arange(eeg.shape[0])/FS_EEG >= start) & (np.arange(eeg.shape[0])/FS_EEG <= end))[0]
        acc_idx = np.where((np.arange(acc.shape[0])/FS_ACC >= start) & (np.arange(acc.shape[0])/FS_ACC <= end))[0]
        gyro_idx = np.where((np.arange(gyro.shape[0])/FS_GYRO >= start) & (np.arange(gyro.shape[0])/FS_GYRO <= end))[0]

        feat = {"period": i+1}

        # EEG features
        if len(eeg_idx) > 0:
            eeg_period = eeg[eeg_idx, :]
            af7 = bandpass_filter(eeg_period[:, 0], FS_EEG)
            af8 = bandpass_filter(eeg_period[:, 1], FS_EEG)
            avg = 0.5*(af7 + af8)
            diff = af7 - af8
            feat.update({
                "af7_std": np.std(af7),
                "af8_std": np.std(af8),
                "avg_rms": rms(avg),
                "avg_line_length": line_length(avg),
                "diff_rms": rms(diff),
                "diff_line_length": line_length(diff),
                "theta_power": bandpower(avg, FS_EEG, 4, 8),
                "alpha_power": bandpower(avg, FS_EEG, 8, 13),
            })

        # ACC features
        if len(acc_idx) > 0:
            acc_period = acc[acc_idx, :]
            for j in range(acc_period.shape[1]):
                feat[f"acc{j+1}_rms"] = rms(acc_period[:, j])
                feat[f"acc{j+1}_line_length"] = line_length(acc_period[:, j])

        # GYRO features
        if len(gyro_idx) > 0:
            gyro_period = gyro[gyro_idx, :]
            for j in range(gyro_period.shape[1]):
                feat[f"gyro{j+1}_rms"] = rms(gyro_period[:, j])
                feat[f"gyro{j+1}_line_length"] = line_length(gyro_period[:, j])

        periods.append(feat)
    return pd.DataFrame(periods)

# ----------------------------
# PROCESS SUBJECT AND LABEL
# ----------------------------
def process_subject(subject_folder, label, trial_type_prefix):
    trial_folders = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if os.path.isdir(os.path.join(subject_folder, f))]
    all_feats = []
    for tf in trial_folders:
        eeg, acc, gyro, markers = load_trial(tf)
        feats = extract_features(eeg, acc, gyro, markers)
        feats["trial"] = os.path.basename(tf)
        feats["label"] = label          # 0 = negative, 1 = fall
        feats["trial_type"] = trial_type_prefix
        all_feats.append(feats)
    return pd.concat(all_feats, ignore_index=True)

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    # Negative trials
    subject_neg = "data/raw/subject01/negative"
    df_neg = process_subject(subject_neg, label=0, trial_type_prefix="negative")
    df_neg.to_csv("features_subject01_negative.csv", index=False)

    # Positive trials
    subject_pos = "data/raw/subject01/positive"
    df_pos = process_subject(subject_pos, label=1, trial_type_prefix="positive")
    df_pos.to_csv("features_subject01_positive.csv", index=False)

    print("Processed features saved!")