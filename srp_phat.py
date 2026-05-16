"""
srp_phat.py — SRP-PHAT Direction of Arrival Estimation
=======================================================
Computes the inter-channel Time Difference of Arrival (TDOA) using
Generalised Cross-Correlation with Phase Transform (GCC-PHAT) weighting,
then converts TDOA to a physical angle.

Reference: Knapp & Carter (1976), IEEE Trans. ASSP, 24(4), 320-327.
"""

import numpy as np
import logging
from typing import Optional
from config import (
    SAMPLE_RATE, MIC_SEPARATION_M, SPEED_OF_SOUND_MS,
    MAX_DELAY_SAMPLES, FFT_SIZE, DOA_SMOOTHING
)

log = logging.getLogger(__name__)


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray,
             fft_size: int = FFT_SIZE,
             max_delay: int = MAX_DELAY_SAMPLES):
    n     = len(sig1)
    n_fft = max(fft_size, 2 * n)

    X1  = np.fft.rfft(sig1, n=n_fft)
    X2  = np.fft.rfft(sig2, n=n_fft)
    XPS = X1 * np.conj(X2)

    mag = np.abs(XPS)
    mag = np.where(mag < 1e-10, 1e-10, mag)
    XPS_phat = XPS / mag

    cc_full = np.fft.irfft(XPS_phat, n=n_fft)
    cc_full = np.concatenate([cc_full[-max_delay:], cc_full[:max_delay + 1]])
    lags    = np.arange(-max_delay, max_delay + 1)
    return cc_full, lags


def tdoa_to_angle_deg(tdoa_samples: int,
                      samplerate: int   = SAMPLE_RATE,
                      mic_sep_m:  float = MIC_SEPARATION_M,
                      c:          float = SPEED_OF_SOUND_MS) -> Optional[float]:
    tau_sec = tdoa_samples / samplerate
    ratio   = (tau_sec * c) / mic_sep_m
    if abs(ratio) > 1.0:
        return None
    return float(np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0))))


def angle_to_sector(angle_deg: Optional[float]) -> str:
    if angle_deg is None:   return "UNKNOWN"
    if   angle_deg < -60:  return "FAR LEFT"
    elif angle_deg < -20:  return "LEFT"
    elif angle_deg < -5:   return "SLIGHT LEFT"
    elif angle_deg <=  5:  return "FRONT / BEHIND"
    elif angle_deg <= 20:  return "SLIGHT RIGHT"
    elif angle_deg <= 60:  return "RIGHT"
    else:                  return "FAR RIGHT"


class SRPPHATProcessor:
    """
    Wraps GCC-PHAT with EMA smoothing on the TDOA estimate.
    """
    def __init__(self, alpha: float = DOA_SMOOTHING, fft_size: int = FFT_SIZE):
        self.alpha    = alpha
        self.fft_size = fft_size
        self._smooth_tdoa: Optional[float] = None

    def update(self, sig_left: np.ndarray, sig_right: np.ndarray) -> Optional[float]:
        if len(sig_left) < 64:
            return None
        cc, lags  = gcc_phat(sig_left, sig_right, fft_size=self.fft_size)
        raw_tdoa  = int(lags[np.argmax(cc)])
        if self._smooth_tdoa is None:
            self._smooth_tdoa = float(raw_tdoa)
        else:
            self._smooth_tdoa = (self.alpha * self._smooth_tdoa
                                 + (1.0 - self.alpha) * raw_tdoa)
        return tdoa_to_angle_deg(round(self._smooth_tdoa))

    def get_full_report(self, sig_left: np.ndarray, sig_right: np.ndarray) -> dict:
        cc, lags = gcc_phat(sig_left, sig_right, fft_size=self.fft_size)
        raw_tdoa = int(lags[np.argmax(cc)])
        angle    = tdoa_to_angle_deg(raw_tdoa)
        return {
            "raw_tdoa_samples":    raw_tdoa,
            "tdoa_microseconds":   (raw_tdoa / SAMPLE_RATE) * 1e6,
            "angle_degrees":       angle,
            "sector":              angle_to_sector(angle),
            "gcc_peak_value":      float(np.max(cc)),
            "gcc_peak_confidence": float(np.max(cc) / (np.mean(np.abs(cc)) + 1e-9)),
        }

    def reset(self):
        self._smooth_tdoa = None
