"""
phone_receiver.py — Receive RPi audio stream and transcribe
============================================================
Run this on the phone (Termux on Android) or laptop for testing.

WiFi mode (default — unchanged):
  pip install faster-whisper numpy
  python3 phone_receiver.py --host 192.168.1.X --port 8888

Bluetooth mode (new):
  pip install faster-whisper numpy pybluez2
  python3 phone_receiver.py --bt --addr DC:A6:32:XX:XX:XX

  --addr  : RPi Bluetooth MAC address (shown at startup or via
            hciconfig / bluetoothctl on the RPi).
            If omitted, the script scans all nearby devices for the
            SmartGlassesAudio service — takes ~10 s.

Both modes use the same protocol: WAV header (44 bytes) then raw
PCM16 mono 16 kHz.  Whisper large-v3 transcribes the same way.

Bluetooth requirements:
  Android/Termux : pkg install bluetooth; pip install pybluez2
                   Grant BLUETOOTH_CONNECT + BLUETOOTH_SCAN permissions
  Linux laptop   : sudo apt-get install libbluetooth-dev
                   pip install pybluez2
"""

import argparse
import logging
import socket
import time
import threading
import numpy as np

log = logging.getLogger("receiver")

SAMPLE_RATE        = 16000
WAV_HDR_SIZE       = 44
MIN_SEC            = 2.0
MAX_SEC            = 8.0
SMART_GLASSES_UUID = "94f39d29-7d6d-437d-973b-fba39e49d4ef"   # must match RPi


class PhoneReceiver:

    def __init__(self, host="", port=8888, model_path="large-v3",
                 bt_mode=False, bt_addr=None):
        self.host        = host
        self.port        = port
        self.bt_mode     = bt_mode
        self.bt_addr     = bt_addr
        self._model      = None
        self._model_path = model_path
        self._buf        = np.zeros(0, dtype=np.float32)
        self._last       = time.time()
        self._jobs: list = []
        self._lock       = threading.Lock()
        self._worker     = threading.Thread(
            target=self._stt_worker, daemon=True, name="stt")

    # ── Whisper STT ───────────────────────────────────────────────────────────

    def _load_model(self):
        log.info("Loading Whisper %s ...", self._model_path)
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self._model_path, device="cpu", compute_type="int8")
        log.info("Whisper loaded.")

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            return ""
        segs, _ = self._model.transcribe(
            audio, beam_size=3, language="en",
            temperature=0.0, vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()

    def _stt_worker(self):
        while True:
            job = None
            with self._lock:
                if self._jobs:
                    job = self._jobs.pop(0)
            if job is not None:
                t0   = time.perf_counter()
                text = self._transcribe(job)
                ms   = (time.perf_counter() - t0) * 1000
                if text:
                    print(f"[{ms:.0f}ms] {text}")
            else:
                time.sleep(0.01)

    def _maybe_flush(self, force=False):
        n       = len(self._buf)
        elapsed = time.time() - self._last
        if n == 0:
            return
        if force or (n >= MIN_SEC * SAMPLE_RATE and elapsed >= MIN_SEC):
            if elapsed >= MAX_SEC or force:
                chunk      = self._buf.copy()
                self._buf  = np.zeros(0, dtype=np.float32)
                self._last = time.time()
                with self._lock:
                    self._jobs.append(chunk)

    # ── Transport A: WiFi TCP (original) ─────────────────────────────────────

    def _connect_tcp(self):
        log.info("Connecting to RPi at %s:%d ...", self.host, self.port)
        sock = socket.create_connection((self.host, self.port), timeout=10)
        log.info("TCP connected.")
        return sock

    # ── Transport B: Bluetooth RFCOMM (new) ───────────────────────────────────

    def _connect_bt(self):
        """
        Discover the SmartGlassesAudio service via SDP, then open an
        RFCOMM socket.  If SDP lookup fails and --addr was given, falls
        back to channel 1 (default RFCOMM channel for SPP devices).
        """
        try:
            import bluetooth
        except ImportError as exc:
            raise ImportError(
                "pybluez2 is required for --bt mode.\n"
                "  pip install pybluez2"
            ) from exc

        addr = self.bt_addr
        log.info(
            "Searching for SmartGlassesAudio BT service%s ...",
            f" on {addr}" if addr else " (scanning nearby devices — ~10 s)",
        )

        services = bluetooth.find_service(
            uuid    = SMART_GLASSES_UUID,
            address = addr if addr else bluetooth.ADDR_ANY,
        )

        if services:
            svc  = services[0]
            host = svc["host"]
            ch   = svc["port"]
            log.info("Found service on %s  channel %d", host, ch)
        elif addr:
            log.warning("SDP lookup failed — falling back to channel 1 on %s", addr)
            host, ch = addr, 1
        else:
            raise RuntimeError(
                "SmartGlassesAudio service not found.\n"
                "  • Make sure the RPi is powered on and discoverable.\n"
                "  • Pass --addr DC:A6:32:XX:XX:XX to skip the scan."
            )

        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((host, ch))
        log.info("Bluetooth RFCOMM connected to %s  ch%d", host, ch)
        return sock

    # ── Shared stream reader ──────────────────────────────────────────────────

    def _read_stream(self, sock):
        """
        Read PCM from an open socket (TCP or BT, same protocol).
        Blocks until the connection drops, then raises an exception
        so the caller can reconnect.
        """
        # Skip WAV header — tells us the format but we already know it
        hdr = b""
        while len(hdr) < WAV_HDR_SIZE:
            chunk = sock.recv(WAV_HDR_SIZE - len(hdr))
            if not chunk:
                raise ConnectionResetError("Stream ended during WAV header")
            hdr += chunk

        chunk_bytes = 512 * 2   # 512 samples × 2 bytes/sample (PCM16)
        while True:
            raw = b""
            while len(raw) < chunk_bytes:
                got = sock.recv(chunk_bytes - len(raw))
                if not got:
                    raise ConnectionResetError("Stream ended")
                raw += got
            pcm = (np.frombuffer(raw, dtype=np.int16)
                   .astype(np.float32) / 32768.0)
            self._buf = np.concatenate([self._buf, pcm])
            self._maybe_flush()

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self):
        self._load_model()
        self._worker.start()

        transport = "Bluetooth" if self.bt_mode else f"WiFi TCP ({self.host}:{self.port})"
        print(f"\n  Smart Glasses receiver — {transport}")
        print("  Connecting to RPi ...\n")

        while True:
            sock = None
            try:
                sock = self._connect_bt() if self.bt_mode else self._connect_tcp()
                print("  ► Streaming — speak near the glasses ...")
                self._read_stream(sock)

            except Exception as exc:
                log.warning("Connection lost: %s — retrying in 2 s ...", exc)
                self._maybe_flush(force=True)
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
                time.sleep(2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Smart Glasses — receive and transcribe RPi audio stream")

    # ── WiFi args (original) ──────────────────────────────────────────────────
    p.add_argument("--host",  default="192.168.1.45",
                   help="RPi IP address  (WiFi mode, default 192.168.1.45)")
    p.add_argument("--port",  type=int, default=8888,
                   help="TCP port        (WiFi mode, default 8888)")

    # ── Bluetooth args (new) ──────────────────────────────────────────────────
    p.add_argument("--bt",    action="store_true",
                   help="Use Bluetooth RFCOMM instead of WiFi TCP")
    p.add_argument("--addr",  default=None, metavar="XX:XX:XX:XX:XX:XX",
                   help="RPi Bluetooth MAC  (BT mode; omit to auto-scan)")

    # ── Whisper ───────────────────────────────────────────────────────────────
    p.add_argument("--model", default="large-v3",
                   help="Whisper model name (default: large-v3)")

    args = p.parse_args()

    PhoneReceiver(
        host       = args.host,
        port       = args.port,
        model_path = args.model,
        bt_mode    = args.bt,
        bt_addr    = args.addr,
    ).run()