"""
main.py — Smart Glasses RPi Zero 2W Audio Pipeline
====================================================

Captures stereo audio from 2× INMP441 microphones, runs:
  DOA estimation (SRP-PHAT) → delay-and-sum beamforming → Wiener
  filter noise reduction
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
import numpy as np

from config import (
    SAMPLE_RATE, CHUNK_FRAMES, DOA_WINDOW_SEC, DOA_SMOOTHING,
    LOG_LEVEL, CALIB_SEC, BEAMFORMER_CONFIDENCE_THRESHOLD,
)
from audio_capture      import DualMicCapture
from srp_phat           import SRPPHATProcessor
from beamformer         import DelayAndSumBeamformer
from noise_reducer      import WienerFilter
from bluetooth_streamer import BluetoothStreamServer

logging.basicConfig(
    level    = getattr(logging, LOG_LEVEL, logging.INFO),
    format   = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("main")


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

        try:
            while self._running:
                left, right = self.capture.read_chunk()
                self._update_doa(left, right)
                mono    = _mix(self.bf, self._use_bf, is_mono, left, right)
                cleaned = self.nr.process(mono)
                if len(cleaned) == 0:
                    continue
                self.server.push(cleaned)

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
