"""
beamformer.py — Delay-and-Sum Beamformer
=========================================
Aligns two microphone channels using the TDOA from GCC-PHAT,
then sums them to spatially filter toward the dominant source.

Coherent (on-axis) signal: combines constructively → +6 dB
Diffuse noise:             combines incoherently  → +3 dB SNR
"""

import numpy as np
import logging
from collections import deque
from config import SAMPLE_RATE, BEAM_GAIN, MAX_DELAY_SAMPLES

log = logging.getLogger(__name__)


class DelayAndSumBeamformer:

    def __init__(self, max_delay: int = MAX_DELAY_SAMPLES, gain: float = BEAM_GAIN):
        self._max_delay = max(max_delay, 4)
        self._gain      = gain
        self._tdoa      = 0.0
        buf = self._max_delay + 8
        self._hist_l = deque([0.0] * buf, maxlen=buf)
        self._hist_r = deque([0.0] * buf, maxlen=buf)

    def set_tdoa(self, tdoa_samples: float) -> None:
        """
        Set the current TDOA (Time Difference of Arrival).
        Positive τ → left mic received sound first → delay left channel.
        Negative τ → right mic received sound first → delay right channel.
        """
        self._tdoa = float(np.clip(tdoa_samples, -self._max_delay, self._max_delay))

    def process(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Delay-align both channels and sum.
        Returns mono float32 array.
        """
        n = len(left)
        self._hist_l.extend(left.tolist())
        self._hist_r.extend(right.tolist())

        hist_l = np.array(self._hist_l)
        hist_r = np.array(self._hist_r)
        delay  = self._tdoa

        if abs(delay) < 0.5:
            out = (hist_l[-n:] + hist_r[-n:]) * 0.5
        elif delay > 0:
            out = self._fractional_delay(hist_l, hist_r, n, delay)
        else:
            out = self._fractional_delay(hist_r, hist_l, n, -delay)

        return np.clip(out * self._gain, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _fractional_delay(early, late, n, delay):
        int_d = int(delay)
        frac  = delay - int_d
        early_delayed = early[-(n + int_d): len(early) - int_d if int_d > 0 else None]
        if len(early_delayed) < n:
            return (early[-n:] + late[-n:]) * 0.5
        if frac > 0 and int_d + 1 < len(early):
            early_next = early[-(n + int_d - 1): len(early) - int_d + 1 if int_d > 1 else None]
            if len(early_next) == n:
                early_delayed = (1 - frac) * early_delayed + frac * early_next
        return (early_delayed[:n] + late[-n:]) * 0.5
