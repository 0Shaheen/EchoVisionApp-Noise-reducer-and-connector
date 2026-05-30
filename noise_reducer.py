"""
noise_reducer.py — Wiener Filter Noise Reducer
===============================================
Per-bin Wiener filter in the STFT domain.

Gain formula: G[k] = SNR[k] / (SNR[k] + 1)

G is always in (0, 1) — can never produce silence.
A hard MAX_REDUCTION_DB cap provides an additional floor.

Noise floor update is gated: only updates during noise-only frames
(frames where total power < 6× noise floor). Speech frames are skipped,
preventing voiced speech from inflating the noise estimate.

Calibration is performed on silence recorded before speech begins.
The median power across calibration frames is used — robust to
accidental transients during the silence window.

For a higher-quality (and slightly heavier) MMSE-LSA enhancer with the
same API, set USE_IMPROVED_NOISE_REDUCER = True in config.py — see
noise_reducer_improved.py.
"""

import numpy as np
import logging
from config import (
    SAMPLE_RATE, NOISE_REDUCE_ENABLED, NOISE_FLOOR_DECAY,
    WIENER_OVERESTIMATE, MAX_REDUCTION_DB, SPECTRAL_FLOOR
)

log = logging.getLogger(__name__)


class WienerFilter:

    SPEECH_FRAME_THRESHOLD = 6.0   # frames with power > 6× noise floor = speech

    def __init__(self,
                 samplerate   = SAMPLE_RATE,
                 frame_ms     = 32,
                 overlap      = 0.5,
                 overestimate = WIENER_OVERESTIMATE,
                 max_atten_db = MAX_REDUCTION_DB,
                 floor        = SPECTRAL_FLOOR,
                 decay        = NOISE_FLOOR_DECAY):

        self.sr           = samplerate
        self.overestimate = overestimate
        self.floor        = floor
        self.max_gain_min = 10 ** (-max_atten_db / 20.0)
        self.decay        = decay
        self.enabled      = NOISE_REDUCE_ENABLED

        self.frame_n = int(samplerate * frame_ms / 1000)
        self.hop_n   = int(self.frame_n * (1 - overlap))
        self.win     = np.hanning(self.frame_n).astype(np.float32)
        self.n_bins  = self.frame_n // 2 + 1

        self._noise_floor = np.ones(self.n_bins, dtype=np.float64) * 1e-12
        self._calibrated  = False
        self._ola_buf     = np.zeros(self.frame_n, dtype=np.float32)
        self._in_buf      = np.zeros(0, dtype=np.float32)

    def calibrate(self, silence_audio: np.ndarray) -> None:
        """
        Seed the noise floor from a recorded silence segment.
        Uses median power across frames — robust to transients.
        """
        n_frames = (len(silence_audio) - self.frame_n) // self.hop_n
        if n_frames < 2:
            log.warning("Calibration segment too short — using auto-init")
            return
        powers = []
        for i in range(n_frames):
            frame = silence_audio[i * self.hop_n: i * self.hop_n + self.frame_n]
            mag   = np.abs(np.fft.rfft(frame * self.win).astype(np.complex128))
            powers.append(mag ** 2)
        self._noise_floor = np.median(np.stack(powers, axis=0), axis=0)
        self._calibrated  = True
        rms    = float(np.sqrt(np.mean(silence_audio ** 2)))
        rms_db = 20.0 * np.log10(rms + 1e-9)
        log.info("Noise floor calibrated: RMS=%.1f dBFS  frames=%d", rms_db, n_frames)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        windowed  = frame * self.win
        spectrum  = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum).astype(np.float64)
        phase     = np.angle(spectrum)
        power     = magnitude ** 2

        # Gate noise floor update: freeze during speech frames
        total_power = float(np.mean(power))
        total_noise = float(np.mean(self._noise_floor)) + 1e-12
        if (total_power / total_noise) < self.SPEECH_FRAME_THRESHOLD:
            self._noise_floor = (self.decay * self._noise_floor
                                 + (1.0 - self.decay) * power)

        # Wiener gain: G[k] = SNR[k] / (SNR[k] + 1)
        eff_noise = self._noise_floor * self.overestimate
        bin_snr   = power / (eff_noise + 1e-12)
        gain      = bin_snr / (bin_snr + 1.0)
        gain      = np.maximum(gain, self.floor)
        gain      = np.maximum(gain, self.max_gain_min)

        clean_spec = (magnitude * gain).astype(np.float32) * np.exp(1j * phase)
        return np.fft.irfft(clean_spec).real.astype(np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio
        self._in_buf = np.concatenate([self._in_buf, audio])
        out_samples  = []
        while len(self._in_buf) >= self.frame_n:
            frame        = self._in_buf[:self.frame_n]
            self._in_buf = self._in_buf[self.hop_n:]
            cleaned      = self._process_frame(frame)
            overlap_len  = self.frame_n - self.hop_n
            self._ola_buf[:overlap_len] += cleaned[:overlap_len]
            out_samples.append(self._ola_buf[:self.hop_n].copy())
            self._ola_buf[:overlap_len] = cleaned[self.hop_n:]
            self._ola_buf[overlap_len:] = 0.0
        if out_samples:
            return np.concatenate(out_samples).astype(np.float32)
        return np.zeros(0, dtype=np.float32)
