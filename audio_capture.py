"""
audio_capture.py — Dual INMP441 I2S Stereo Capture
====================================================
Captures 24-bit stereo audio from two INMP441 microphones over I2S.
Returns normalised float32 arrays in [-1.0, +1.0].

INMP441 I2S format:
  - 32-bit I2S frames, audio in upper 24 bits (bits 31:8), lower 8 bits = 0
  - Left  mic: L/R pin → GND   → I2S channel 0 (WS = LOW)
  - Right mic: L/R pin → 3.3V  → I2S channel 1 (WS = HIGH)
"""

import queue
import logging
import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE, CHANNELS, CAPTURE_DTYPE,
    INMP441_BIT_SHIFT, CHUNK_FRAMES, ALSA_DEVICE
)

log = logging.getLogger(__name__)

_INT24_MAX = float(2 ** 23)


def inmp441_to_float32(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw int32 INMP441 samples → normalised float32 [-1, 1].
    The INMP441 packs 24-bit audio into the upper 24 bits of a 32-bit frame.
    """
    return (raw >> INMP441_BIT_SHIFT).astype(np.int32).astype(np.float32) / _INT24_MAX


class DualMicCapture:
    """
    Thread-safe stereo I2S capture from two INMP441 microphones.

    Usage:
        cap = DualMicCapture()
        cap.start()
        left, right = cap.read_chunk()   # float32 arrays
        cap.stop()
    """

    def __init__(self, device=ALSA_DEVICE, samplerate=SAMPLE_RATE, chunk=CHUNK_FRAMES):
        self.device     = device
        self.samplerate = samplerate
        self.chunk      = chunk
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self._stream    = None
        self._running   = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("sounddevice: %s", status)
        self._q.put(indata.copy())   # indata shape: (chunk, 2) int32

    def start(self):
        log.info("Opening INMP441 I2S: device=%s  %d Hz  int32",
                 self.device, self.samplerate)
        self._stream = sd.InputStream(
            device=self.device,
            channels=CHANNELS,
            samplerate=self.samplerate,
            blocksize=self.chunk,
            dtype=CAPTURE_DTYPE,
            callback=self._callback,
        )
        self._running = True
        self._stream.start()
        log.info("I2S stream running — Left=ch0 (L/R→GND)  Right=ch1 (L/R→3.3V)")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        log.info("I2S stream stopped.")

    def read_chunk(self, timeout=1.0):
        """Returns (left, right) as float32 arrays of length chunk."""
        try:
            raw = self._q.get(timeout=timeout)   # (chunk, 2) int32
        except queue.Empty:
            log.warning("Audio read timeout — returning silence")
            s = np.zeros(self.chunk, dtype=np.float32)
            return s, s
        return inmp441_to_float32(raw[:, 0]), inmp441_to_float32(raw[:, 1])

    def read_seconds(self, duration_sec: float):
        """Record duration_sec of audio and return (left, right) float32 arrays."""
        n = max(1, int(np.ceil(duration_sec * self.samplerate / self.chunk)))
        ls, rs = [], []
        for _ in range(n):
            l, r = self.read_chunk()
            ls.append(l); rs.append(r)
        return np.concatenate(ls), np.concatenate(rs)

    @property
    def is_running(self):
        return self._running


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import sounddevice as sd
    print("\nAvailable audio devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] >= 2:
            print(f"  [{i:2d}] {dev['name']}  ch={dev['max_input_channels']}")

    cap = DualMicCapture()
    cap.start()
    print("Recording 3 s — speak near the microphones ...")
    left, right = cap.read_seconds(3.0)
    cap.stop()
    rms_l = float(np.sqrt(np.mean(left  ** 2)))
    rms_r = float(np.sqrt(np.mean(right ** 2)))
    print(f"Left  mic RMS = {rms_l:.5f}  {'OK' if rms_l > 1e-4 else 'SILENT — check wiring'}")
    print(f"Right mic RMS = {rms_r:.5f}  {'OK' if rms_r > 1e-4 else 'SILENT — check wiring'}")
