import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import resolve_byprop, StreamInlet
from scipy.signal import butter, filtfilt, find_peaks


# ----------------------------
# Helpers
# ----------------------------
def connect_stream(stream_type: str, timeout: float = 5.0) -> StreamInlet:
    streams = resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        raise RuntimeError(f"No LSL stream found for type={stream_type}")
    inlet = StreamInlet(streams[0], max_buflen=10)
    print(f"Connected to {stream_type}: {streams[0].name()} ({streams[0].channel_count()} ch)")
    return inlet


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 2) -> np.ndarray:
    if len(x) < max(10, order * 3):
        return x
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if high_n >= 1.0:
        high_n = 0.99
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, x)


def estimate_bpm_from_ppg(ppg_signal: np.ndarray, fs: float) -> float | None:
    if len(ppg_signal) < int(fs * 4):
        return None

    try:
        # Rough human HR band: 42–180 BPM
        filt = bandpass_filter(ppg_signal, fs, 0.7, 3.0, order=2)
    except Exception:
        return None

    distance = max(1, int(0.4 * fs))  # at most ~150 BPM
    prominence = max(1e-6, np.std(filt) * 0.25)

    peaks, _ = find_peaks(filt, distance=distance, prominence=prominence)
    if len(peaks) < 2:
        return None

    ibi_sec = np.diff(peaks) / fs
    mean_ibi = np.mean(ibi_sec)
    if mean_ibi <= 0:
        return None

    bpm = 60.0 / mean_ibi
    if 35 <= bpm <= 220:
        return float(bpm)
    return None


# ----------------------------
# Main dashboard
# ----------------------------
class MuseDashboard:
    def __init__(self):
        # Connect streams
        self.eeg_inlet = connect_stream("EEG")
        self.gyro_inlet = connect_stream("GYRO")
        self.acc_inlet = connect_stream("ACC")
        self.ppg_inlet = connect_stream("PPG")

        # Sampling assumptions / buffers
        self.eeg_fs = 256.0
        self.motion_fs = 52.0
        self.ppg_fs = 64.0  # approximate; enough for live visualization/BPM

        self.eeg_seconds = 5
        self.motion_seconds = 8
        self.ppg_seconds = 10

        self.eeg_n = int(self.eeg_fs * self.eeg_seconds)
        self.motion_n = int(self.motion_fs * self.motion_seconds)
        self.ppg_n = int(self.ppg_fs * self.ppg_seconds)

        # Muse EEG often comes as 5 channels in your stream
        self.eeg_channels = 5
        self.gyro_channels = 3
        self.acc_channels = 3
        self.ppg_channels = 3

        self.eeg_buf = np.zeros((self.eeg_n, self.eeg_channels), dtype=float)
        self.gyro_buf = np.zeros((self.motion_n, self.gyro_channels), dtype=float)
        self.acc_buf = np.zeros((self.motion_n, self.acc_channels), dtype=float)
        self.ppg_buf = np.zeros((self.ppg_n, self.ppg_channels), dtype=float)

        self.bpm_history = deque(maxlen=120)
        self.time_history = deque(maxlen=120)

        # GUI
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title="Muse Live Dashboard")
        self.win.resize(1400, 900)

        pg.setConfigOptions(antialias=True)

        # EEG plot
        self.eeg_plot = self.win.addPlot(row=0, col=0, colspan=2, title="EEG (stacked)")
        self.eeg_plot.showGrid(x=True, y=True)
        self.eeg_plot.addLegend()
        self.eeg_curves = [
            self.eeg_plot.plot(pen=pg.mkPen(width=1), name=f"EEG {i}")
            for i in range(self.eeg_channels)
        ]

        # Gyro plot
        self.gyro_plot = self.win.addPlot(row=1, col=0, title="Gyro")
        self.gyro_plot.showGrid(x=True, y=True)
        self.gyro_plot.addLegend()
        self.gyro_curves = [
            self.gyro_plot.plot(pen=pg.mkPen(width=2), name=label)
            for label in ["GX", "GY", "GZ"]
        ]

        # Acc plot
        self.acc_plot = self.win.addPlot(row=1, col=1, title="Accelerometer")
        self.acc_plot.showGrid(x=True, y=True)
        self.acc_plot.addLegend()
        self.acc_curves = [
            self.acc_plot.plot(pen=pg.mkPen(width=2), name=label)
            for label in ["AX", "AY", "AZ"]
        ]

        # PPG plot
        self.ppg_plot = self.win.addPlot(row=2, col=0, title="PPG")
        self.ppg_plot.showGrid(x=True, y=True)
        self.ppg_plot.addLegend()
        self.ppg_curves = [
            self.ppg_plot.plot(pen=pg.mkPen(width=2), name=label)
            for label in ["PPG1", "PPG2", "PPG3"]
        ]

        # BPM plot
        self.bpm_plot = self.win.addPlot(row=2, col=1, title="Estimated Heart Rate (BPM)")
        self.bpm_plot.showGrid(x=True, y=True)
        self.bpm_curve = self.bpm_plot.plot(pen=pg.mkPen(width=3))

        self.bpm_label = pg.LabelItem(justify="left")
        self.win.addItem(self.bpm_label, row=3, col=0, colspan=2)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.last_bpm = None

    def _append_sample(self, buf: np.ndarray, sample: list[float]) -> None:
        sample_arr = np.asarray(sample, dtype=float)
        if sample_arr.shape[0] != buf.shape[1]:
            # pad or trim if stream shape differs
            out = np.zeros(buf.shape[1], dtype=float)
            n = min(len(sample_arr), buf.shape[1])
            out[:n] = sample_arr[:n]
            sample_arr = out
        buf[:-1] = buf[1:]
        buf[-1] = sample_arr

    def pull_all_available(self, inlet: StreamInlet, target_buf: np.ndarray, max_reads: int = 64):
        reads = 0
        while reads < max_reads:
            sample, _ = inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            self._append_sample(target_buf, sample)
            reads += 1

    def update(self):
        # Pull latest samples
        self.pull_all_available(self.eeg_inlet, self.eeg_buf, max_reads=128)
        self.pull_all_available(self.gyro_inlet, self.gyro_buf, max_reads=64)
        self.pull_all_available(self.acc_inlet, self.acc_buf, max_reads=64)
        self.pull_all_available(self.ppg_inlet, self.ppg_buf, max_reads=64)

        # EEG stacked display
        eeg_x = np.arange(self.eeg_n) / self.eeg_fs
        eeg = self.eeg_buf.copy()

        # crude centering / stacking
        offsets = np.arange(self.eeg_channels) * 150.0
        for i in range(self.eeg_channels):
            y = eeg[:, i] - np.mean(eeg[:, i]) + offsets[i]
            self.eeg_curves[i].setData(eeg_x, y)

        # Gyro
        motion_x = np.arange(self.motion_n) / self.motion_fs
        for i in range(self.gyro_channels):
            self.gyro_curves[i].setData(motion_x, self.gyro_buf[:, i])

        # Acc
        for i in range(self.acc_channels):
            self.acc_curves[i].setData(motion_x, self.acc_buf[:, i])

        # PPG
        ppg_x = np.arange(self.ppg_n) / self.ppg_fs
        for i in range(self.ppg_channels):
            self.ppg_curves[i].setData(ppg_x, self.ppg_buf[:, i])

        # BPM from first PPG channel
        ppg_primary = self.ppg_buf[:, 0]
        bpm = estimate_bpm_from_ppg(ppg_primary, self.ppg_fs)
        now = time.time()
        if bpm is not None:
            self.last_bpm = bpm
            self.bpm_history.append(bpm)
            self.time_history.append(now)

        if len(self.bpm_history) > 1:
            t0 = self.time_history[0]
            bpm_t = np.array([t - t0 for t in self.time_history], dtype=float)
            bpm_y = np.array(self.bpm_history, dtype=float)
            self.bpm_curve.setData(bpm_t, bpm_y)

        if self.last_bpm is not None:
            self.bpm_label.setText(f"<span style='font-size:18pt'>Estimated BPM: <b>{self.last_bpm:.1f}</b></span>")
        else:
            self.bpm_label.setText("<span style='font-size:18pt'>Estimated BPM: <b>calculating...</b></span>")

    def run(self):
        self.win.show()
        sys.exit(self.app.exec())


if __name__ == "__main__":
    dash = MuseDashboard()
    dash.run()