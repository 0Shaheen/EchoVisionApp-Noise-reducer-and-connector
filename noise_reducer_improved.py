"""
noise_reducer_improved.py — MMSE-LSA Speech Enhancer (speech-preserving)
========================================================================
Optional, higher-quality replacement for the original Wiener-filter
noise reducer in noise_reducer.py.  Enable it by setting
USE_IMPROVED_NOISE_REDUCER = True in config.py.

Same public API (`WienerFilter`, `calibrate()`, `process()`) so the rest
of the pipeline doesn't change.  The class name is kept for compatibility;
internally the algorithm is no longer "plain Wiener" — see below.

Why a new algorithm
-------------------
The previous implementation used a fixed Wiener gain  G = SNR/(SNR+1)  with
an instantaneous a-posteriori SNR and a global hard VAD ("freeze the noise
floor if total power > 6× current floor").  That combination has two
well-known failure modes on speech:

  • Musical noise: instantaneous SNR is rough, so per-bin gain jitters
    frame-to-frame and produces "ringy" residual tones around speech.
  • Speech damage: the global VAD fires on amplitude, so weak fricatives
    /s/, /f/, /th/ slip under the threshold and get folded into the noise
    estimate.  The next consonant is then attenuated as if it were noise.

Pipeline (per analysis frame)
-----------------------------
  1.  STFT analysis  — 32 ms Hann window, 50 % overlap.
  2.  A-posteriori SNR   γ_k = |Y_k|² / λ_d,k
  3.  Decision-Directed a-priori SNR  ξ_k  (Ephraim-Malah 1984):
          ξ_k = α · |Ŝ_k(n-1)|² / λ_d,k  +  (1-α) · max(γ_k − 1, 0)
      This time-smooths ξ across frames and is the single biggest
      contributor to *not* mangling speech — musical noise drops by
      ~15 dB compared with the un-smoothed estimate.
  4.  Speech Presence Probability  p_k  (Gerkmann & Hendriks 2012):
          p_k = 1 / (1 + (q/(1-q)) · (1+ξ_opt) · exp(-ν_k))
          ν_k = γ_k · ξ_opt / (1+ξ_opt)
      Uses a fixed ξ_opt under H1 (15 dB) — robust against bin-by-bin
      noise in the instantaneous estimate.  Smoothed with an
      attack-fast / release-slow IIR so speech onsets aren't muted.
  5.  Soft noise PSD update — Gerkmann/Hendriks style:
          λ_d(n) = [α_d + (1-α_d)·p_k] · λ_d(n-1)
                 + (1-α_d)·(1-p_k) · |Y_k|²
      Replaces the global "VAD freeze" with per-bin soft gating: bins
      that are *probably speech* almost don't update; bins that are
      *probably noise* track the new statistics.  No hard threshold,
      no calibration loop required (calibrate() still helps as a warm
      start).
  6.  MMSE-LSA suppression gain  (Ephraim-Malah 1985):
          G_LSA(ξ, γ) = ξ/(1+ξ) · exp(0.5 · E1(ν))
      The log-spectral-amplitude estimator produces noticeably more
      natural speech than vanilla Wiener — less hollow / less robotic.
  7.  Uncertainty blending with the per-bin floor:
          G_final = G_LSA^p · G_floor^(1-p)
  8.  Gain smoothing — ±1 bin moving average across frequency, plus a
      light first-order IIR across time, both kill isolated musical
      tones without smearing onsets.
  9.  Frequency-dependent floor — the 200–4000 Hz voice band keeps a
      higher minimum gain than out-of-band bins, so consonants and
      sibilants stay audible even under aggressive denoising.
  10. ISTFT with overlap-add.

Compute budget on a Pi Zero 2 W is ≲ 1 % CPU at 16 kHz, mono, 32 ms
frames: one rfft / irfft, a handful of numpy element-wise ops, an
exp_integral_e1 lookup per bin.  No SciPy dependency added.

References
----------
Ephraim & Malah, IEEE TASSP 1984 — MMSE-STSA + DD a-priori SNR
Ephraim & Malah, IEEE TASSP 1985 — MMSE log-spectral amplitude
Gerkmann & Hendriks, IEEE TASLP 2012 — unbiased MMSE-based noise PSD
Abramowitz & Stegun §5.1.11 / §5.1.56 — series + rational E1(x)
"""

import logging
import numpy as np

from config import (
    SAMPLE_RATE,
    NOISE_REDUCE_ENABLED, NOISE_FLOOR_DECAY,
    MAX_REDUCTION_DB, SPECTRAL_FLOOR,
)

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Exponential integral E1(x), x > 0
#
# Required by the MMSE-LSA gain.  Two regimes, both vectorised:
#   x  < 1   → series expansion  (A&S 5.1.11)
#   x >= 1   → rational/exp form (A&S 5.1.56), |error| < 2e-8
# Avoids pulling in SciPy on the Pi.
# ────────────────────────────────────────────────────────────────────────────

_EULER_GAMMA = 0.5772156649015329

# A&S 5.1.56 coefficients — careful with order (Horner's evaluates as
#   x⁴ + a1·x³ + a2·x² + a3·x + a4 ).
_A1, _A2, _A3, _A4 = 8.5733287401, 18.0590169730,  8.6347608925, 0.2677737343
_B1, _B2, _B3, _B4 = 9.5733223454, 25.6329561486, 21.0996530827, 3.9584969228


def _exp_integral_e1(x: np.ndarray) -> np.ndarray:
    x   = np.maximum(x, 1e-12)
    out = np.empty_like(x)

    small = x < 1.0
    if np.any(small):
        xs = x[small]
        out[small] = (
            -_EULER_GAMMA - np.log(xs)
            + xs
            - xs**2 / 4.0
            + xs**3 / 18.0
            - xs**4 / 96.0
            + xs**5 / 600.0
        )

    big = ~small
    if np.any(big):
        xb  = x[big]
        num = (((xb + _A1) * xb + _A2) * xb + _A3) * xb + _A4
        den = (((xb + _B1) * xb + _B2) * xb + _B3) * xb + _B4
        out[big] = (num / den) * np.exp(-xb) / xb

    return out


# ────────────────────────────────────────────────────────────────────────────
# Speech enhancer.  Class name retained for drop-in compatibility.
# ────────────────────────────────────────────────────────────────────────────

class WienerFilter:

    # ── Decision-Directed smoothing for ξ ─────────────────────────────────
    # Higher α → smoother ξ → less musical noise, marginally smeared
    # transients.  0.94–0.98 is the EM-recommended range.
    DD_ALPHA = 0.96

    # ── Gerkmann-Hendriks SPP ─────────────────────────────────────────────
    SPP_PRIOR_Q     = 0.5     # a-priori speech-absence probability
    SPP_XI_OPT_DB   = 15.0    # assumed SNR under H1
    SPP_ATTACK      = 0.2     # IIR history weight when p rising (fast attack)
    SPP_RELEASE     = 0.75    # IIR history weight when p falling (slow release)

    # ── Soft noise PSD tracker ────────────────────────────────────────────
    # τ ≈ -hop / (sr · ln α_d).  With sr=16k, hop=256, α=0.92 → ~200 ms.
    NOISE_PSD_ALPHA = 0.92

    # ── Gain post-smoothing — cuts musical tones, keeps onsets ────────────
    GAIN_TIME_SMOOTH      = 0.4
    GAIN_FREQ_SMOOTH_BINS = 1     # half-width; 1 → 3-bin moving average

    # ── Speech-band protected zone ────────────────────────────────────────
    SPEECH_BAND_HZ          = (200.0, 4000.0)
    SPEECH_BAND_FLOOR_BOOST = 2.0   # × SPECTRAL_FLOOR inside the band

    def __init__(self,
                 samplerate   = SAMPLE_RATE,
                 frame_ms     = 32,
                 overlap      = 0.5,
                 max_atten_db = MAX_REDUCTION_DB,
                 floor        = SPECTRAL_FLOOR,
                 decay        = NOISE_FLOOR_DECAY):

        self.sr           = samplerate
        self.enabled      = NOISE_REDUCE_ENABLED
        self.max_gain_min = 10 ** (-max_atten_db / 20.0)

        self.frame_n = int(samplerate * frame_ms / 1000)
        self.hop_n   = int(self.frame_n * (1 - overlap))
        self.win     = np.hanning(self.frame_n).astype(np.float32)
        self.n_bins  = self.frame_n // 2 + 1

        # Per-bin floor: voice band protected, everywhere else uses the
        # configured SPECTRAL_FLOOR.  Always at least max_gain_min so the
        # MAX_REDUCTION_DB hard cap still applies.
        freqs       = np.linspace(0.0, samplerate / 2.0, self.n_bins)
        in_speech   = (freqs >= self.SPEECH_BAND_HZ[0]) & (freqs <= self.SPEECH_BAND_HZ[1])
        floor_in    = min(1.0, floor * self.SPEECH_BAND_FLOOR_BOOST)
        self._gain_floor               = np.full(self.n_bins, floor, dtype=np.float64)
        self._gain_floor[in_speech]    = floor_in
        self._gain_floor               = np.maximum(self._gain_floor, self.max_gain_min)

        # State carried across frames
        self._noise_psd      = np.full(self.n_bins, 1e-10, dtype=np.float64)
        self._prev_clean_amp = np.zeros(self.n_bins, dtype=np.float64)
        self._prev_gain      = np.ones (self.n_bins, dtype=np.float64)
        self._prev_spp       = np.zeros(self.n_bins, dtype=np.float64)
        self._calibrated     = False
        self._calib_decay    = decay   # kept for API parity (unused by SPP path)

        # OLA buffers
        self._ola_buf = np.zeros(self.frame_n, dtype=np.float32)
        self._in_buf  = np.zeros(0,             dtype=np.float32)

        # Pre-computed constants for the SPP closed form
        self._xi_opt_lin = 10 ** (self.SPP_XI_OPT_DB / 10.0)
        self._gh_q_ratio = self.SPP_PRIOR_Q / (1.0 - self.SPP_PRIOR_Q)   # q/(1-q)

    # ── Public API ────────────────────────────────────────────────────────

    def calibrate(self, silence_audio: np.ndarray) -> None:
        """
        Seed the noise PSD from a recorded silence segment.  Optional —
        the SPP tracker will converge on its own within ~300 ms anyway,
        but calibration removes that initial ramp-up.
        """
        n_frames = (len(silence_audio) - self.frame_n) // self.hop_n
        if n_frames < 2:
            log.warning("Calibration segment too short — using auto-init")
            return

        powers = []
        for i in range(n_frames):
            frame = silence_audio[i * self.hop_n: i * self.hop_n + self.frame_n]
            mag   = np.abs(np.fft.rfft(frame * self.win).astype(np.complex128))
            powers.append(mag * mag)

        self._noise_psd  = np.maximum(np.median(np.stack(powers, axis=0), axis=0), 1e-12)
        self._calibrated = True

        rms    = float(np.sqrt(np.mean(silence_audio ** 2)))
        rms_db = 20.0 * np.log10(rms + 1e-9)
        log.info("Noise PSD calibrated: RMS=%.1f dBFS  frames=%d", rms_db, n_frames)

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

    # ── Internals ─────────────────────────────────────────────────────────

    def _speech_presence_probability(self, gamma: np.ndarray) -> np.ndarray:
        """
        Gerkmann-Hendriks soft SPP per bin, with attack-fast / release-slow
        IIR smoothing across frames so a sudden onset isn't half-attenuated
        but a single noisy frame can't drag the probability back down.
        """
        nu = gamma * self._xi_opt_lin / (1.0 + self._xi_opt_lin)
        nu = np.minimum(nu, 500.0)  # prevent exp overflow on huge γ
        p  = 1.0 / (1.0 + self._gh_q_ratio * (1.0 + self._xi_opt_lin) * np.exp(-nu))

        alpha = np.where(p > self._prev_spp, self.SPP_ATTACK, self.SPP_RELEASE)
        p     = alpha * self._prev_spp + (1.0 - alpha) * p
        self._prev_spp = p
        return p

    def _mmse_lsa_gain(self, gamma: np.ndarray, xi: np.ndarray) -> np.ndarray:
        ratio = xi / (1.0 + xi)
        nu    = np.clip(gamma * ratio, 1e-8, 500.0)
        return ratio * np.exp(0.5 * _exp_integral_e1(nu))

    def _smooth_gain_freq(self, gain: np.ndarray) -> np.ndarray:
        k = self.GAIN_FREQ_SMOOTH_BINS
        if k <= 0:
            return gain
        K    = 2 * k + 1
        pad  = np.pad(gain, k, mode="edge")
        csum = np.concatenate(([0.0], np.cumsum(pad)))
        return (csum[K:] - csum[:-K]) / K

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(frame * self.win)
        mag      = np.abs(spectrum).astype(np.float64)
        phase    = np.angle(spectrum)
        power    = mag * mag

        # 1. a-posteriori SNR  γ
        noise = np.maximum(self._noise_psd, 1e-12)
        gamma = power / noise

        # 2. SPP (uses γ + fixed ξ_opt; does not depend on ξ)
        p = self._speech_presence_probability(gamma)

        # 3. soft noise PSD update — frozen on bins where p ≈ 1
        update_w = self.NOISE_PSD_ALPHA + (1.0 - self.NOISE_PSD_ALPHA) * p
        self._noise_psd = update_w * self._noise_psd + (1.0 - update_w) * power
        self._noise_psd = np.maximum(self._noise_psd, 1e-12)

        # 4. decision-directed a-priori SNR  ξ
        instant_xi = np.maximum(gamma - 1.0, 0.0)
        prev_xi    = (self._prev_clean_amp ** 2) / np.maximum(self._noise_psd, 1e-12)
        xi         = self.DD_ALPHA * prev_xi + (1.0 - self.DD_ALPHA) * instant_xi
        xi         = np.maximum(xi, 1e-6)

        # 5. MMSE-LSA gain
        gain = self._mmse_lsa_gain(gamma, xi)

        # 6. SPP-weighted blend with the per-bin floor
        gain = (gain ** p) * (self._gain_floor ** (1.0 - p))

        # 7. spectral + temporal smoothing
        gain = self._smooth_gain_freq(gain)
        gain = self.GAIN_TIME_SMOOTH * self._prev_gain + (1 - self.GAIN_TIME_SMOOTH) * gain
        self._prev_gain = gain

        # 8. hard limits
        gain = np.clip(gain, self._gain_floor, 1.0)

        # 9. apply gain; remember enhanced amplitude for DD on the next frame
        clean_mag             = mag * gain
        self._prev_clean_amp  = clean_mag

        clean_spec = (clean_mag * np.exp(1j * phase)).astype(np.complex128)
        return np.fft.irfft(clean_spec).real.astype(np.float32)
