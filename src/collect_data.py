# src/collect_data.py

from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import time
import os

# ----------------------------
# PROMPT FOR SUBJECT
# ----------------------------
subject_id = input("Enter subject ID (e.g., subject01): ").strip()

# Base folders
RAW_FOLDER = f'data/raw/{subject_id}'
NEGATIVE_FOLDER = os.path.join(RAW_FOLDER, "negative")
POSITIVE_FOLDER = os.path.join(RAW_FOLDER, "positive")
os.makedirs(NEGATIVE_FOLDER, exist_ok=True)
os.makedirs(POSITIVE_FOLDER, exist_ok=True)

# ----------------------------
# CONFIG
# ----------------------------
NUM_PERIODS = 10          # periods per trial
PERIOD_DURATION = 5       # seconds
DELAY_BEFORE_START = 2    # seconds
PULL_TIMEOUT = 0.1        # seconds for pull_sample

# ----------------------------
# DEFINE TRIALS
# ----------------------------
negative_trials = [
    {"name": "sitting", "instruction": "Sit normally"},
    {"name": "standing", "instruction": "Stand still"}
]

positive_trials = [
    {"name": "trial01_forward_crouch", "instruction": "Fall forward + crouch"},
    {"name": "trial02_twist_right", "instruction": "Fall + twist to right side"},
    {"name": "trial03_twist_left", "instruction": "Fall + twist to left side"},
    {"name": "trial04_backward_crouch", "instruction": "Fall backward + crouch"}
]

# ----------------------------
# RESOLVE STREAMS
# ----------------------------
def wait_for_stream(stream_type, timeout=5):
    inlet = None
    while inlet is None:
        streams = resolve_byprop('type', stream_type, timeout=timeout)
        if streams:
            inlet = StreamInlet(streams[0])
            print(f"{stream_type} stream found: {streams[0].name()}")
        else:
            print(f"Waiting for {stream_type} stream...")
            time.sleep(1)
    return inlet

print("Connecting to streams...")
eeg_inlet = wait_for_stream('EEG')
acc_inlet = wait_for_stream('ACC')
gyro_inlet = wait_for_stream('GYRO')

# ----------------------------
# RECORD A SINGLE TRIAL
# ----------------------------
def record_trial(trial_name, trial_instruction, folder):
    eeg_data, acc_data, gyro_data, timestamps = [], [], [], []
    markers = []

    print(f"\n--- Starting trial: {trial_name} ---")
    print(f"Instruction: {trial_instruction}")

    period_count = 0
    while period_count < NUM_PERIODS:
        print(f"\nPeriod {period_count + 1}/{NUM_PERIODS}")
        input("Press Enter to start this period...")

        # Start period
        print(f"Waiting {DELAY_BEFORE_START} seconds before recording...")
        time.sleep(DELAY_BEFORE_START)

        start_time = time.time()
        markers.append({"timestamp": start_time, "marker": "period_start"})
        print(f"Recording {PERIOD_DURATION} seconds...")

        # Record data
        while time.time() - start_time < PERIOD_DURATION:
            eeg_sample, ts_eeg = eeg_inlet.pull_sample(timeout=PULL_TIMEOUT)
            acc_sample, ts_acc = acc_inlet.pull_sample(timeout=PULL_TIMEOUT)
            gyro_sample, ts_gyro = gyro_inlet.pull_sample(timeout=PULL_TIMEOUT)

            if eeg_sample:
                eeg_data.append(eeg_sample)
                timestamps.append(ts_eeg)
            if acc_sample:
                acc_data.append(acc_sample)
            if gyro_sample:
                gyro_data.append(gyro_sample)  # append sample, not timestamp

        markers.append({"timestamp": time.time(), "marker": "period_end"})
        print("Period complete!")
        period_count += 1

    # ----------------------------
    # SAVE CSV FILES
    # ----------------------------
    def save_csv(data_list, filename, col_prefix):
        if len(data_list) == 0:
            print(f"Warning: no data collected for {filename}, skipping CSV.")
            return
        df = pd.DataFrame(data_list, columns=[f'{col_prefix}{i+1}' for i in range(len(data_list[0]))])
        df.to_csv(filename, index=False)

    save_csv(eeg_data, f'{folder}/{trial_name}_eeg.csv', 'EEG')
    save_csv(acc_data, f'{folder}/{trial_name}_acc.csv', 'ACC')
    save_csv(gyro_data, f'{folder}/{trial_name}_gyro.csv', 'GYRO')
    pd.DataFrame(markers).to_csv(f'{folder}/{trial_name}_markers.csv', index=False)

    print(f"Trial {trial_name} saved successfully!")
    print(f"EEG samples: {len(eeg_data)}, ACC samples: {len(acc_data)}, GYRO samples: {len(gyro_data)}")

# ----------------------------
# RUN NEGATIVE TRIALS
# ----------------------------
for trial in negative_trials:
    record_trial(trial["name"], trial["instruction"], NEGATIVE_FOLDER)

# ----------------------------
# RUN POSITIVE TRIALS
# ----------------------------
for trial in positive_trials:
    record_trial(trial["name"], trial["instruction"], POSITIVE_FOLDER)

print("\nAll trials complete!")