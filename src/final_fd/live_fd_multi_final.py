import os
import time
import subprocess
import numpy as np
import pandas as pd
import joblib
from pylsl import StreamInlet, resolve_byprop

# ---------------------------
# CONFIG
# ---------------------------

MODEL_FILE = "fall_detector_rf_multi.pkl"

FS_ACC = 64
FS_EEG = 256

WINDOW_SIZE_SEC = 1.0
WINDOW_OVERLAP = 0.5

EEG_WEIGHT = 1.5

CONSECUTIVE_WINDOWS = 2
FALL_CONFIRMATION_COUNT = 3

COOLDOWN_TIME = 30  # seconds

BLINK_SCRIPT = "blink_detection.py"

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------

def extract_features(acc_win, gyro_win, eeg_win, eeg_weight=EEG_WEIGHT):

    features = []

    # ACC features
    acc_mean = np.mean(acc_win, axis=0)
    acc_std = np.std(acc_win, axis=0)
    acc_mag = np.sqrt(np.sum(acc_win**2, axis=1)).mean()

    features.extend(acc_mean)
    features.extend(acc_std)
    features.append(acc_mag)

    # Gyro features
    gyro_mean = np.mean(gyro_win, axis=0)
    gyro_std = np.std(gyro_win, axis=0)
    gyro_mag = np.sqrt(np.sum(gyro_win**2, axis=1)).mean()

    features.extend(gyro_mean)
    features.extend(gyro_std)
    features.append(gyro_mag)

    # EEG features
    eeg_mean = np.mean(eeg_win, axis=0) * eeg_weight
    eeg_std = np.std(eeg_win, axis=0) * eeg_weight

    features.extend(eeg_mean)
    features.extend(eeg_std)

    return np.array(features)

# ---------------------------
# WINDOWING
# ---------------------------

def sliding_windows(signal, fs, window_sec=1.0, overlap=0.5):

    window_samples = int(fs * window_sec)
    step = int(window_samples * (1 - overlap))

    windows = []

    for start in range(0, len(signal) - window_samples + 1, step):
        end = start + window_samples
        windows.append(signal[start:end])

    return windows

# ---------------------------
# LOAD MODEL
# ---------------------------

clf = joblib.load(MODEL_FILE)

print(f"Loaded model: {MODEL_FILE}")

# ---------------------------
# STREAM CONNECTION
# ---------------------------

def wait_for_stream(stream_type, timeout=5):

    inlet = None

    while inlet is None:

        streams = resolve_byprop("type", stream_type, timeout=timeout)

        if streams:
            inlet = StreamInlet(streams[0])
            print(f"{stream_type} stream found: {streams[0].name()}")

        else:
            print(f"Waiting for {stream_type} stream...")
            time.sleep(1)

    return inlet

print("Connecting to Muse streams...")

eeg_inlet = wait_for_stream("EEG")
acc_inlet = wait_for_stream("ACC")
gyro_inlet = wait_for_stream("GYRO")

# ---------------------------
# BUFFERS
# ---------------------------

acc_buffer = []
gyro_buffer = []
eeg_buffer = []

# ---------------------------
# LOG FILE
# ---------------------------

log_file = "live_fall_log.csv"

if not os.path.exists(log_file):

    pd.DataFrame(columns=["timestamp","fall_detected"]).to_csv(log_file,index=False)

# ---------------------------
# STATE VARIABLES
# ---------------------------

consec_count = 0
confirmed_falls = 0
cooldown_until = 0

# ---------------------------
# BLINK TRIGGER
# ---------------------------

def run_blink_check():

    print("\nLaunching blink detection...\n")

    subprocess.run(["python", BLINK_SCRIPT])

    print("\nBlink check finished\n")

# ---------------------------
# LIVE LOOP
# ---------------------------

print("\nStarting live fall detection...\n")

try:

    while True:

        # Cooldown check
        if time.time() < cooldown_until:
            continue

        # Pull samples
        acc_sample, _ = acc_inlet.pull_sample(timeout=0.1)
        gyro_sample, _ = gyro_inlet.pull_sample(timeout=0.1)
        eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.1)

        if acc_sample is not None:
            acc_buffer.append(acc_sample)

        if gyro_sample is not None:
            gyro_buffer.append(gyro_sample)

        if eeg_sample is not None:
            eeg_buffer.append(eeg_sample)

        # Check window availability
        if len(acc_buffer) >= int(FS_ACC * WINDOW_SIZE_SEC) and len(eeg_buffer) >= int(FS_EEG * WINDOW_SIZE_SEC):

            acc_win = np.array(acc_buffer[-int(FS_ACC * WINDOW_SIZE_SEC):])
            gyro_win = np.array(gyro_buffer[-int(FS_ACC * WINDOW_SIZE_SEC):])
            eeg_win = np.array(eeg_buffer[-int(FS_EEG * WINDOW_SIZE_SEC):])

            feat = extract_features(acc_win, gyro_win, eeg_win)

            pred = clf.predict([feat])[0]

            if pred == 1:
                consec_count += 1
            else:
                consec_count = 0

            # Confirm fall
            if consec_count >= CONSECUTIVE_WINDOWS:

                confirmed_falls += 1

                print(f"FALL DETECTED ({confirmed_falls}/{FALL_CONFIRMATION_COUNT})")

                pd.DataFrame([[time.time(),1]],
                             columns=["timestamp","fall_detected"]
                             ).to_csv(log_file,mode="a",header=False,index=False)

                consec_count = 0

                # Trigger blink system
                if confirmed_falls >= FALL_CONFIRMATION_COUNT:

                    print("\n*** FALL EVENT CONFIRMED ***")

                    run_blink_check()

                    confirmed_falls = 0

                    cooldown_until = time.time() + COOLDOWN_TIME

                    print(f"Cooling down for {COOLDOWN_TIME} seconds...\n")

            else:

                print("stable")

                pd.DataFrame([[time.time(),0]],
                             columns=["timestamp","fall_detected"]
                             ).to_csv(log_file,mode="a",header=False,index=False)

except KeyboardInterrupt:

    print("\nStopping live fall detection...")