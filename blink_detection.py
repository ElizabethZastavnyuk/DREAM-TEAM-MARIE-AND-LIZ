import numpy as np
import time
import tkinter as tk
from tkinter import font as tkfont
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


# ── GUI ──────────────────────────────────────────────────────────────────────

class WelfarePopup:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Status Check")
        self.root.geometry("480x340")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        self.root.attributes("-topmost", True)

        self.title_font  = tkfont.Font(family="Helvetica", size=15, weight="bold")
        self.body_font   = tkfont.Font(family="Helvetica", size=12)
        self.status_font = tkfont.Font(family="Helvetica", size=11, slant="italic")
        self.btn_font    = tkfont.Font(family="Helvetica", size=12, weight="bold")

        self.recording = False
        self._build_waiting_screen()

    def _build_waiting_screen(self):
        for w in self.root.winfo_children():
            w.destroy()

        tk.Label(self.root, text="⚠️  Unstable State Detected",
                 bg="#1a1a2e", fg="#e94560",
                 font=self.title_font).pack(pady=(30, 8))

        tk.Label(self.root, text="Are you okay?",
                 bg="#1a1a2e", fg="#ffffff",
                 font=self.body_font).pack()

        tk.Label(self.root,
                 text="Blink TWICE if you are okay\nBlink THREE TIMES if you need help",
                 bg="#1a1a2e", fg="#a8b2d8",
                 font=self.body_font, justify="center").pack(pady=(12, 16))

        self.start_btn = tk.Button(
            self.root,
            text="▶  Start Recording",
            bg="#e94560", fg="#000000",
            font=self.btn_font,
            relief="flat", padx=16, pady=8,
            cursor="hand2",
            command=self._on_start
        )
        self.start_btn.pack(pady=(0, 16))

        self.status_label = tk.Label(self.root, text="Press the button when ready",
                                     bg="#1a1a2e", fg="#4fc3f7",
                                     font=self.status_font)
        self.status_label.pack()

        self.blink_label = tk.Label(self.root, text="",
                                    bg="#1a1a2e", fg="#ffffff",
                                    font=self.status_font)
        self.blink_label.pack(pady=(4, 0))

    def _on_start(self):
        self.recording = True
        self.start_btn.config(state="disabled", text="Recording...")
        self.status_label.config(text="Listening for blinks...")
        self.blink_label.config(text="Blinks detected: 0")

    def update_blink_count(self, count):
        self.blink_label.config(text=f"Blinks detected: {count}")
        self.root.update()

    def show_result(self, blink_count):
        for w in self.root.winfo_children():
            w.destroy()

        if blink_count == 2:
            msg   = "✅  Thank you, stay safe!"
            color = "#00e676"
        elif blink_count == 3:
            msg   = "🚨  Calling for help!"
            color = "#e94560"
        else:
            msg   = f"Response unclear ({blink_count} blinks).\nPlease try again."
            color = "#ffb300"

        tk.Label(self.root, text=msg,
                 bg="#1a1a2e", fg=color,
                 font=tkfont.Font(family="Helvetica", size=16, weight="bold"),
                 wraplength=420, justify="center").pack(expand=True)

        self.root.after(4000, self.root.destroy)
        self.root.mainloop()

    def start(self):
        self.root.update()


popup = WelfarePopup()
popup.start()

# ── Wait for button press ─────────────────────────────────────────────────────

print("Waiting for user to press Start Recording...")
while not popup.recording:
    popup.root.update()
    time.sleep(0.05)

print("Recording started!")

# ── Blink detection loop ──────────────────────────────────────────────────────

RECORD_SECONDS = 5

start_time = time.time()
blink_count = 0
last_blink_time = 0

while time.time() - start_time < RECORD_SECONDS:
    popup.root.update()

    eeg_data, timestamp = inlet.pull_chunk(
        timeout=1, max_samples=int(SHIFT_LENGTH * fs))

    if len(eeg_data) == 0:
        continue

    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

    eeg_buffer, filter_state = utils.update_buffer(
        eeg_buffer, ch_data, notch=True, filter_state=filter_state)

    data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
    band_powers = utils.compute_band_powers(data_epoch, fs)
    delta_val = band_powers[Band.Delta]

    if delta_val >= 1:
        if time.time() - last_blink_time > 0.3:
            blink_count += 1
            last_blink_time = time.time()
            popup.update_blink_count(blink_count)

popup.show_result(blink_count)