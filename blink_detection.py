import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
import utils


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

INDEX_CHANNEL = [0]


print("Looking for EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    raise RuntimeError("Can't find EEG stream.")

print("Connected to EEG stream")

inlet = StreamInlet(streams[0], max_chunklen=12)
info = inlet.info()

fs = int(info.nominal_srate())

eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
filter_state = None

print("\nUnstable state detected. Are you okay?")
print("Blink TWICE if you are okay, blink THREE TIMES if you need help.")
print("\nRecording your response...")

start_time = time.time()
blink_count = 0
last_blink_time = 0

while time.time() - start_time < 3:

    eeg_data, timestamp = inlet.pull_chunk(
        timeout=1, max_samples=int(SHIFT_LENGTH * fs))

    if len(eeg_data) == 0:
        continue

    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

    eeg_buffer, filter_state = utils.update_buffer(
        eeg_buffer,
        ch_data,
        notch=True,
        filter_state=filter_state
    )

    data_epoch = utils.get_last_data(
        eeg_buffer,
        EPOCH_LENGTH * fs
    )

    band_powers = utils.compute_band_powers(data_epoch, fs)

    delta_val = band_powers[Band.Delta]

    if delta_val >= 1:
        if time.time() - last_blink_time > 0.3:
            blink_count += 1
            last_blink_time = time.time()
            print("Blink detected")

print("\nRecording finished")

if blink_count == 2:
    print("Thank you, stay safe!")
elif blink_count == 3:
    print("Calling for help!")
else:
    print(f"Response unclear ({blink_count} blinks detected). Please try again.")
