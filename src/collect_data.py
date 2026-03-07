# src/collect_data.py

from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import time
import os

# ----------------------------
# PROMPT FOR SUBJECT AND TRIAL
# ----------------------------
subject_id = input("Enter subject ID (e.g., subject01): ").strip()
trial_id = input("Enter trial ID (e.g., trial01): ").strip()

# ----------------------------
# CONFIG
# ----------------------------
DURATION_SEC = 10               # seconds per trial
DATA_FOLDER = f'data/raw/{subject_id}'
os.makedirs(DATA_FOLDER, exist_ok=True)

# ----------------------------
# RESOLVE STREAMS
# ----------------------------
print("Looking for EEG stream...")
eeg_stream = resolve_byprop('type', 'EEG', timeout=5)[0]
print("Looking for ACC stream...")
acc_stream = resolve_byprop('type', 'ACC', timeout=5)[0]
print("Looking for GYRO stream...")
gyro_stream = resolve_byprop('type', 'GYRO', timeout=5)[0]

# ----------------------------
# CREATE INLETS
# ----------------------------
eeg_inlet = StreamInlet(eeg_stream)
acc_inlet = StreamInlet(acc_stream)
gyro_inlet = StreamInlet(gyro_stream)

# ----------------------------
# RECORD DATA
# ----------------------------
eeg_data, acc_data, gyro_data, timestamps = [], [], [], []

print(f"Recording {trial_id} for {DURATION_SEC} seconds...")
start_time = time.time()
while time.time() - start_time < DURATION_SEC:
    eeg_sample, ts_eeg = eeg_inlet.pull_sample(timeout=0.0)
    acc_sample, ts_acc = acc_inlet.pull_sample(timeout=0.0)
    gyro_sample, ts_gyro = gyro_inlet.pull_sample(timeout=0.0)
    
    if eeg_sample:
        eeg_data.append(eeg_sample)
        timestamps.append(ts_eeg)
    if acc_sample:
        acc_data.append(acc_sample)
    if gyro_sample:
        gyro_data.append(gyro_sample)

# ----------------------------
# SAVE CSV FILES
# ----------------------------
# EEG
eeg_df = pd.DataFrame(eeg_data, columns=[f'EEG{i+1}' for i in range(len(eeg_data[0]))])
eeg_df.to_csv(f'{DATA_FOLDER}/{trial_id}_eeg.csv', index=False)

# ACC
acc_df = pd.DataFrame(acc_data, columns=[f'ACC{i+1}' for i in range(len(acc_data[0]))])
acc_df.to_csv(f'{DATA_FOLDER}/{trial_id}_acc.csv', index=False)

# GYRO
gyro_df = pd.DataFrame(gyro_data, columns=[f'GYRO{i+1}' for i in range(len(gyro_data[0]))])
gyro_df.to_csv(f'{DATA_FOLDER}/{trial_id}_gyro.csv', index=False)

# TIMESTAMPS / MARKERS
timestamps_df = pd.DataFrame(timestamps, columns=['timestamp'])
timestamps_df.to_csv(f'{DATA_FOLDER}/{trial_id}_markers.csv', index=False)

print(f"{trial_id} saved successfully in {DATA_FOLDER}!")