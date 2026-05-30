"""
main.py — Smart Glasses RPi Zero 2W Audio Pipeline
====================================================

Captures stereo audio from 2× INMP441 microphones, runs:
  DOA estimation (SRP-PHAT) → delay-and-sum beamforming → noise
  reduction (Wiener filter, or the MMSE-LSA enhancer when
  USE_IMPROVED_NOISE_REDUCER is enabled in config.py)
and streams the processed 16 kHz mono PCM16 audio to the EchoVision
phone app over Bluetooth Classic RFCOMM.

Start manually:
  python3 main.py

Start on boot:
  sudo systemctl start smartglasses

Stop:
  Ctrl+C  or  sudo systemctl stop smartglasses
"""

import logging
import signal
import sys
import time
import numpy as np

from config import (
    SAMPLE_RATE, CHUNK_FRAMES, DOA_WINDOW_SEC, DOA_SMOOTHING,
    LOG_LEVEL, CALIB_SEC, BEAMFORMER_CONFIDENCE_THRESHOLD,
    USE_IMPROVED_NOISE_REDUCER,
)
from audio_capture      import DualMicCapture
from srp_phat           import SRPPHATProcessor
from beamformer         import DelayAndSumBeamformer
# Pick the noise reducer at import time. Both modules expose WienerFilter
# with an identical calibrate()/process() API, so the rest of the pipeline
# is unaffected.  Flip USE_IMPROVED_NOISE_REDUCER in config.py to switch.
if USE_IMPROVED_NOISE_REDUCER:
    from noise_reducer_improved import WienerFilter   # MMSE-LSA speech enhancer
else:
    from noise_reducer          import WienerFilter   # original Wiener filter
from bluetooth_streamer import BluetoothStreamServer

logging.basicConfig(
    level    = getattr(logging, LOG_LEVEL, logging.INFO),
    format   = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("main")
log.info("Noise reducer: %s",
         "MMSE-LSA (improved)" if USE_IMPROVED_NOISE_REDUCER else "Wiener (original)")


def _mix(bf, use_bf, is_mono, l, r):
    """Return mono signal using the appropriate strategy."""
    if is_mono:  return l.copy()
    if use_bf:   return bf.process(l, r)
    return (l + r) * 0.5


class SmartGlassesPipeline:

    def __init__(self):
        self.capture  = DualMicCapture()
        self.doa      = SRPPHATProcessor()
        self.bf       = DelayAndSumBeamformer()
        self.nr       = WienerFilter()
        self.server   = BluetoothStreamServer()
        self._running = False

        self._doa_l: list[np.ndarray] = []
        self._doa_r: list[np.ndarray] = []
        self._doa_samps = int(DOA_WINDOW_SEC * SAMPLE_RATE)
        self._tdoa      = 0.0
        self._use_bf    = False

    # ── Startup calibration ───────────────────────────────────────────────────

    def _calibrate(self) -> bool:
        print()
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │  STAY SILENT for {CALIB_SEC:.0f} seconds (noise calibration)  │")
        print(f"  └─────────────────────────────────────────────────┘")

        left, right = self.capture.read_seconds(CALIB_SEC)
        is_mono = np.allclose(left, right, atol=1e-6)

        report  = self.doa.get_full_report(left, right)
        conf    = report.get("gcc_peak_confidence", 0.0)
        use_bf  = (not is_mono) and (conf >= BEAMFORMER_CONFIDENCE_THRESHOLD)

        print(f"  Stereo confidence : {conf:.2f}  →  "
              f"Beamformer: {'ON' if use_bf else 'OFF (mono device)'}")

        chunk = CHUNK_FRAMES
        calib_chunks = []
        for i in range(0, len(left), chunk):
            l = left [i: i + chunk]
            r = right[i: i + chunk]
            if len(l) < chunk:
                l = np.pad(l, (0, chunk - len(l)))
                r = np.pad(r, (0, chunk - len(r)))
            calib_chunks.append(_mix(self.bf, use_bf, is_mono, l, r))

        calib_mono = np.concatenate(calib_chunks)
        self.bf  = DelayAndSumBeamformer()
        self.nr.calibrate(calib_mono)

        print(f"  ✓  Calibration complete")
        return use_bf

    # ── DOA update ────────────────────────────────────────────────────────────

    def _update_doa(self, left, right):
        self._doa_l.append(left)
        self._doa_r.append(right)
        if sum(len(a) for a in self._doa_l) < self._doa_samps:
            return
        l_win = np.concatenate(self._doa_l)[-self._doa_samps:]
        r_win = np.concatenate(self._doa_r)[-self._doa_samps:]
        rep   = self.doa.get_full_report(l_win, r_win)
        self._tdoa = (DOA_SMOOTHING * self._tdoa
                      + (1 - DOA_SMOOTHING) * rep["raw_tdoa_samples"])
        self.bf.set_tdoa(self._tdoa)
        keep = self._doa_samps // 2
        self._doa_l = [l_win[-keep:]]
        self._doa_r = [r_win[-keep:]]
        log.debug("DOA %.1f° (%s)  conf=%.2f",
                  rep.get("angle_degrees") or 0,
                  rep.get("sector", "?"),
                  rep.get("gcc_peak_confidence", 0))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        stream_info = self.server.start()

        print()
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║       Smart Glasses — Audio Stream Server        ║")
        print("  ╠══════════════════════════════════════════════════╣")
        print(f"  ║  Transport : Bluetooth RFCOMM                    ║")
        print(f"  ║  {stream_info:<48} ║")
        print("  ║                                                  ║")
        print("  ║  Pair the EchoVision phone app to receive audio  ║")
        print("  ║  (16 kHz mono PCM16)                             ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print()

        self.capture.start()

        is_mono  = False
        self._use_bf = self._calibrate()

        print()
        print("  ► Streaming — waiting for phone to connect ...")
        print("  ► Press Ctrl+C to stop")
        print()

        self._running = True

        # --- DIAGNOSTIC: per-stage timing, logged once per second ---
        t_acc = {"read": 0.0, "doa": 0.0, "mix": 0.0, "nr": 0.0, "push": 0.0, "n": 0}
        t_log = time.monotonic()

        try:
            while self._running:
                _a = time.monotonic()
                left, right = self.capture.read_chunk()
                _b = time.monotonic()
                self._update_doa(left, right)
                _c = time.monotonic()
                mono    = _mix(self.bf, self._use_bf, is_mono, left, right)
                _d = time.monotonic()
                cleaned = self.nr.process(mono)
                _e = time.monotonic()

                t_acc["read"] += _b - _a
                t_acc["doa"]  += _c - _b
                t_acc["mix"]  += _d - _c
                t_acc["nr"]   += _e - _d
                t_acc["n"]    += 1

                if len(cleaned) != 0:
                    self.server.push(cleaned)
                    t_acc["push"] += time.monotonic() - _e

                now = time.monotonic()
                if now - t_log >= 1.0:
                    n = max(1, t_acc["n"])
                    log.info("[TIMING] iters=%d/s  read=%.1f  doa=%.1f  mix=%.1f  "
                             "nr=%.1f  push=%.1f  ms/iter",
                             t_acc["n"],
                             1000 * t_acc["read"] / n, 1000 * t_acc["doa"] / n,
                             1000 * t_acc["mix"]  / n, 1000 * t_acc["nr"]  / n,
                             1000 * t_acc["push"] / n)
                    t_acc = {"read": 0.0, "doa": 0.0, "mix": 0.0, "nr": 0.0, "push": 0.0, "n": 0}
                    t_log = now

        except KeyboardInterrupt:
            print()
            log.info("Stopped by user.")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.capture.stop()
        self.server.stop()
        log.info("Pipeline stopped.")


def main():
    pipeline = SmartGlassesPipeline()

    def _shutdown(sig, frame):
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    pipeline.run()


if __name__ == "__main__":
    main()
