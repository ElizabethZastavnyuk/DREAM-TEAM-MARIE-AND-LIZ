from pylsl import resolve_byprop, StreamInlet
import time

stream_types = ["EEG", "GYRO", "ACC", "PPG"]
inlets = {}

for stype in stream_types:
    streams = resolve_byprop('type', stype, timeout=5)
    if streams:
        inlets[stype] = StreamInlet(streams[0])
        print(f"Connected to {stype}: {streams[0].name()} ({streams[0].channel_count()} channels)")
    else:
        print(f"No stream found for {stype}")

print("\nStreaming live samples...\n")

while True:
    for stype, inlet in inlets.items():
        sample, timestamp = inlet.pull_sample(timeout=0.0)
        if sample is not None:
            print(f"{stype:5s} | {timestamp:.3f} | {sample}")
    time.sleep(0.05)