"""
bluetooth_streamer.py — Bluetooth RFCOMM Audio Stream Server (stdlib)
====================================================================
Streams processed mono PCM16 audio to the EchoVision phone app over
Bluetooth Classic RFCOMM, using ONLY the Python standard library
(the `socket` module with AF_BLUETOOTH / BTPROTO_RFCOMM).  There is NO
pybluez / pybluez2 dependency, so this works on Python 3.13 with no
C-extension to compile.

----------------------------------------------------------------------
Wire framing (binary-safe)
----------------------------------------------------------------------
The Android client (react-native-bluetooth-classic) reads the RFCOMM
stream as newline-delimited text and STRIPS the delimiter.  Sending raw
PCM would let any 0x0A ("\\n") sample byte be silently deleted, shifting
byte alignment and turning speech into noise.  To stay binary-safe each
audio chunk is framed as:

    base64(pcm16_le_bytes) + b"\\n"

base64 uses only 7-bit ASCII (A-Z a-z 0-9 + / =), so it never contains
0x0A; the trailing newline is an unambiguous frame separator.  The phone
splits on "\\n", base64-decodes each frame back to PCM16, and feeds it to
Whisper.  Overhead is ~33% (≈43 KB/s) — well within RFCOMM's ~700 kbps.

----------------------------------------------------------------------
SDP service discovery (one-time Pi setup)
----------------------------------------------------------------------
The Android app connects using the standard Serial Port Profile (SPP)
UUID 00001101-0000-1000-8000-00805F9B34FB.  Android resolves the SPP
UUID to an RFCOMM channel via an SDP record on the Pi.  The stdlib
socket module cannot create SDP records, so we register one with the
`sdptool` utility, which requires BlueZ running in compatibility mode.

Enable it once (only needed if the phone can't discover the channel):
  1. Add --compat to the bluetoothd ExecStart line, e.g.:
       /etc/systemd/system/bluetooth.service.d/override.conf
         [Service]
         ExecStart=
         ExecStart=/usr/libexec/bluetooth/bluetoothd -E --compat
  2. sudo systemctl daemon-reload
  3. sudo systemctl restart bluetooth

start() calls `sdptool add` for you automatically; if compat mode is off
the call is skipped with a warning and the phone falls back to channel 1.

Pair the phone once via bluetoothctl, then it reconnects on every boot.
"""

import base64
import logging
import queue
import socket
import subprocess
import threading
import time
import numpy as np

from config import BT_SERVICE_NAME, BT_RFCOMM_CHANNEL

log = logging.getLogger(__name__)

# Bind to any local Bluetooth adapter.  Provided by CPython's socket
# module when built with Bluetooth support; fall back to the literal.
BDADDR_ANY = getattr(socket, "BDADDR_ANY", "00:00:00:00:00:00")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _register_sdp_spp(channel: int) -> bool:
    """
    Best-effort: publish a Serial Port Profile SDP record on `channel`
    via `sdptool` so Android can resolve the SPP UUID to our channel.
    Requires BlueZ compat mode (-C / --compat).  Failure is non-fatal.
    """
    try:
        subprocess.run(
            ["sdptool", "add", f"--channel={channel}", "SP"],
            check=True, capture_output=True, timeout=5,
        )
        log.info("SDP: Serial Port Profile registered on RFCOMM channel %d", channel)
        return True
    except FileNotFoundError:
        log.warning("SDP: 'sdptool' not found — install the 'bluez' package")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="ignore").strip()
        log.warning("SDP: sdptool failed (%s). Enable BlueZ compat mode "
                    "(add --compat to bluetoothd).", detail or exc.returncode)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("SDP: registration skipped (%s)", exc)
    return False


# ── Server ────────────────────────────────────────────────────────────────────

class BluetoothStreamServer:
    """
    Bluetooth RFCOMM server — streams processed mono PCM16 audio as
    newline-delimited base64 frames (see module docstring).

    Listens on an RFCOMM channel and publishes an SPP SDP record so the
    phone can find it.  Accepts one client at a time; if the phone
    disconnects it waits silently and reconnects on the next accept().
    push() discards the oldest chunk rather than blocking if the send
    queue is full.

    Usage (API identical to the previous versions):
        srv = BluetoothStreamServer()
        info = srv.start()   # returns a human-readable status string
        srv.push(audio)      # float32 numpy array in [-1, 1]
        srv.stop()
    """

    def __init__(self):
        self._server_sock = None
        self._client_sock = None
        self._client_lock = threading.Lock()
        self._send_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._running = False
        self._channel = BT_RFCOMM_CHANNEL
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="bt-accept")
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="bt-send")

        # --- DIAGNOSTIC: throughput counters (samples/sec produced vs sent) ---
        self._diag_pushed = 0     # samples handed to push() in the last window
        self._diag_sent = 0       # samples actually written to the socket
        self._diag_dropped = 0    # frames dropped because the queue was full
        self._diag_last = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> str:
        """Open the RFCOMM server socket, register SDP, start the threads."""
        if not hasattr(socket, "AF_BLUETOOTH"):
            raise RuntimeError(
                "This Python build has no AF_BLUETOOTH support — cannot open "
                "a Bluetooth socket.  Use the system Python on Raspberry Pi OS."
            )

        self._server_sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        try:
            self._server_sock.bind((BDADDR_ANY, BT_RFCOMM_CHANNEL))
        except OSError as exc:
            # Configured channel busy — let the kernel pick a free one.
            log.warning("RFCOMM channel %d unavailable (%s) — auto-selecting",
                        BT_RFCOMM_CHANNEL, exc)
            self._server_sock.bind((BDADDR_ANY, 0))

        self._server_sock.listen(1)
        self._channel = self._server_sock.getsockname()[1]

        _register_sdp_spp(self._channel)

        self._running = True
        self._accept_thread.start()
        self._send_thread.start()

        return (f"Bluetooth RFCOMM  ch={self._channel}  "
                f"service={BT_SERVICE_NAME!r}  (SPP / base64 frames / stdlib socket)")

    def stop(self):
        self._running = False
        with self._client_lock:
            if self._client_sock:
                try:
                    self._client_sock.close()
                except Exception:
                    pass
                self._client_sock = None
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

    def push(self, audio: np.ndarray):
        """
        Push a float32 chunk to the BT stream.
        If no phone is connected the PCM bytes are silently discarded.
        If the queue is full the oldest chunk is dropped to prevent lag.
        """
        self._diag_pushed += len(audio)
        pcm = _float32_to_pcm16(audio)
        try:
            self._send_queue.put_nowait(pcm)
        except queue.Full:
            self._diag_dropped += 1
            try:    self._send_queue.get_nowait()
            except Exception: pass
            try:    self._send_queue.put_nowait(pcm)
            except Exception: pass

    # ── Internal threads ──────────────────────────────────────────────────────

    def _accept_loop(self):
        """Block on accept() and register the new client. base64 frames follow."""
        while self._running:
            try:
                self._server_sock.settimeout(1.0)
                conn, info = self._server_sock.accept()
            except socket.timeout:
                continue   # re-check _running and loop
            except OSError:
                if self._running:
                    continue
                break

            conn.settimeout(None)
            addr = info[0] if isinstance(info, (tuple, list)) else info
            log.info("Phone connected via BT: %s", addr)
            print(f"\n  +  Phone connected via Bluetooth: {addr}")

            # No up-front header: newline-delimited base64 PCM frames follow.
            with self._client_lock:
                if self._client_sock:
                    try:
                        self._client_sock.close()
                    except Exception:
                        pass
                self._client_sock = conn

    def _send_loop(self):
        """Dequeue PCM bytes, base64-frame them, and forward to the phone."""
        while self._running:
            try:
                pcm = self._send_queue.get(timeout=0.05)
            except queue.Empty:
                pcm = None

            if pcm is not None:
                with self._client_lock:
                    client = self._client_sock
                if client is not None:
                    try:
                        # Binary-safe frame: base64 payload + newline separator.
                        client.sendall(base64.b64encode(pcm) + b"\n")
                        self._diag_sent += len(pcm) // 2
                    except OSError:
                        log.info("BT link dropped — waiting for phone to reconnect ...")
                        print("  x  BT link dropped — waiting for phone to reconnect ...")
                        with self._client_lock:
                            if self._client_sock is client:
                                try:
                                    client.close()
                                except Exception:
                                    pass
                                self._client_sock = None

            # --- DIAGNOSTIC: log produced-vs-sent throughput once per second ---
            now = time.monotonic()
            if self._diag_last is None:
                self._diag_last = now
            elif now - self._diag_last >= 1.0:
                log.info("[RATE] produced=%d samp/s  sent=%d samp/s  "
                         "dropped=%d frames/s  queue=%d/64",
                         self._diag_pushed, self._diag_sent,
                         self._diag_dropped, self._send_queue.qsize())
                self._diag_pushed = 0
                self._diag_sent = 0
                self._diag_dropped = 0
                self._diag_last = now
