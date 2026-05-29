"""
bluetooth_streamer.py — Bluetooth RFCOMM Audio Stream Server (stdlib)
====================================================================
Streams processed mono PCM16 audio to the EchoVision phone app over
Bluetooth Classic RFCOMM, using ONLY the Python standard library
(the `socket` module with AF_BLUETOOTH / BTPROTO_RFCOMM).  There is NO
pybluez / pybluez2 dependency, so this works out of the box on
Python 3.13 with no C-extension to compile.

Streaming protocol (unchanged):
  [44 bytes]  WAV header  (16 kHz, mono, PCM16)
  [inf bytes] Raw int16 PCM — continuous stream

Bandwidth: 16 kHz x 2 bytes x 1 ch ~= 32 KB/s  (~256 kbps).
BT Classic RFCOMM supports ~700 kbps — comfortable headroom.

----------------------------------------------------------------------
SDP service discovery (one-time Pi setup)
----------------------------------------------------------------------
The Android app connects using the standard Serial Port Profile (SPP)
UUID 00001101-0000-1000-8000-00805F9B34FB.  Android resolves the SPP
UUID to an RFCOMM channel via an SDP record on the Pi.  The stdlib
socket module cannot create SDP records, so we register one with the
`sdptool` utility, which requires BlueZ running in compatibility mode.

Enable it once:
  1. Edit the bluetooth unit and add --compat to the daemon line:
       sudoedit /lib/systemd/system/bluetooth.service
     Change the ExecStart line to (path may be /usr/libexec/... ):
       ExecStart=/usr/libexec/bluetooth/bluetoothd --compat
  2. sudo systemctl daemon-reload
  3. sudo systemctl restart bluetooth

start() then calls `sdptool add` for you automatically.  If compat mode
is off the call is skipped with a warning; pairing/streaming may still
work if the phone falls back to RFCOMM channel 1.

Pair the phone once via bluetoothctl, then it reconnects on every boot.
"""

import io
import logging
import queue
import socket
import struct
import subprocess
import threading
import wave
import numpy as np

from config import (
    BT_SERVICE_NAME, BT_RFCOMM_CHANNEL,
    OUT_SAMPLE_RATE, OUT_CHANNELS, OUT_BITS,
)

log = logging.getLogger(__name__)

# Bind to any local Bluetooth adapter.  Provided by CPython's socket
# module when built with Bluetooth support; fall back to the literal.
BDADDR_ANY = getattr(socket, "BDADDR_ANY", "00:00:00:00:00:00")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _make_wav_header() -> bytes:
    """
    44-byte WAV header with data size = 0xFFFFFFFF (streaming / unknown).
    Identical to the original — any client that handled that stream works
    here without modification.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(OUT_CHANNELS)
        wf.setsampwidth(OUT_BITS // 8)
        wf.setframerate(OUT_SAMPLE_RATE)
        wf.setnframes(0)
    hdr = bytearray(buf.getvalue())
    struct.pack_into("<I", hdr,  4, 0xFFFFFFFF)
    struct.pack_into("<I", hdr, 40, 0xFFFFFFFF)
    return bytes(hdr)


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
    Bluetooth RFCOMM server — streams processed mono PCM16 audio.

    Listens on an RFCOMM channel and publishes an SPP SDP record so the
    phone can find it.  Accepts one client at a time; if the phone
    disconnects it waits silently and reconnects on the next accept().
    push() discards the oldest chunk rather than blocking if the send
    queue is full.

    Usage (API identical to the previous pybluez2 version):
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
                f"service={BT_SERVICE_NAME!r}  (SPP / stdlib socket)")

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
        pcm = _float32_to_pcm16(audio)
        try:
            self._send_queue.put_nowait(pcm)
        except queue.Full:
            try:    self._send_queue.get_nowait()
            except Exception: pass
            try:    self._send_queue.put_nowait(pcm)
            except Exception: pass

    # ── Internal threads ──────────────────────────────────────────────────────

    def _accept_loop(self):
        """Block on accept(), send the WAV header, register the new client."""
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

            # Send the WAV header so the phone can skip it once per connection.
            try:
                conn.sendall(_make_wav_header())
            except OSError:
                try:
                    conn.close()
                except Exception:
                    pass
                continue

            with self._client_lock:
                if self._client_sock:
                    try:
                        self._client_sock.close()
                    except Exception:
                        pass
                self._client_sock = conn

    def _send_loop(self):
        """Dequeue PCM bytes and forward them to the connected phone."""
        while self._running:
            try:
                pcm = self._send_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._client_lock:
                client = self._client_sock
            if client is None:
                continue   # no phone connected — discard

            try:
                client.sendall(pcm)
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
