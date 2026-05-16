"""
bluetooth_streamer.py — Bluetooth RFCOMM Audio Stream Server
============================================================
Drop-in replacement for audio_streamer.py using Bluetooth Classic
RFCOMM instead of WiFi TCP.  The streaming protocol is identical:
  [44 bytes]  WAV header  (16 kHz, mono, PCM16)
  [∞ bytes]   Raw int16 PCM — continuous stream

The RPi registers an SDP service named "SmartGlassesAudio" so the
phone can discover it by name rather than a hard-coded channel number.

BlueZ device class is configured externally (see BT_SETUP section in
SETUP_GUIDE.txt) to 0x200448:
  Major service  : Audio  (bit 21)
  Major device   : Audio/Video  (0x04 << 8)
  Minor device   : Hearing Aid  (0x12 << 2 = 0x48)
→ Android surfaces the paired RPi under
  Settings → Accessibility → Hearing aids in the Bluetooth menu.

Bandwidth: 16 kHz × 2 bytes × 1 ch ≈ 32 KB/s  (~256 kbps)
BT Classic RFCOMM supports ~700 kbps — comfortable headroom.

Pair once via bluetoothctl (see SETUP_GUIDE.txt §BT), then the phone
reconnects automatically on every boot.

Phone side:
  python3 phone_receiver.py --bt --addr DC:A6:32:XX:XX:XX
"""

import io
import logging
import queue
import struct
import threading
import wave
import numpy as np

try:
    import bluetooth
except ImportError as exc:
    raise ImportError(
        "pybluez2 is required for Bluetooth mode.\n"
        "  RPi  : pip install pybluez2\n"
        "  Phone: pip install pybluez2\n"
        "  Also : sudo apt-get install bluetooth libbluetooth-dev"
    ) from exc

from config import (
    BT_SERVICE_NAME, BT_RFCOMM_CHANNEL,
    OUT_SAMPLE_RATE, OUT_CHANNELS, OUT_BITS,
)

log = logging.getLogger(__name__)

# Fixed UUID for SDP advertisement — must match phone_receiver.py
# Generated once; do NOT change after the first pairing.
SMART_GLASSES_UUID = "94f39d29-7d6d-437d-973b-fba39e49d4ef"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _make_wav_header() -> bytes:
    """
    44-byte WAV header with data size = 0xFFFFFFFF (streaming / unknown).
    Identical to the TCP version — any client that handles that stream
    works here without modification.
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


# ── Server ────────────────────────────────────────────────────────────────────

class BluetoothStreamServer:
    """
    Bluetooth RFCOMM server — streams processed mono PCM16 audio.

    Registers "SmartGlassesAudio" via SDP.  Accepts one client at a
    time; if the phone disconnects it waits silently and reconnects on
    the next accept().  Push() discards the oldest chunk rather than
    blocking if the send queue is full.

    Usage (mirrors AudioStreamServer):
        srv = BluetoothStreamServer()
        url = srv.start()   # returns "Bluetooth RFCOMM  addr=...  ch=..."
        srv.push(audio)     # float32 numpy array
        srv.stop()
    """

    def __init__(self):
        self._server_sock  = None
        self._client_sock  = None
        self._client_lock  = threading.Lock()
        self._send_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._running      = False
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="bt-accept")
        self._send_thread   = threading.Thread(
            target=self._send_loop,  daemon=True, name="bt-send")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> str:
        """Open RFCOMM server socket, advertise via SDP, start threads."""
        self._server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self._server_sock.bind(("", BT_RFCOMM_CHANNEL))
        self._server_sock.listen(1)
        actual_ch = self._server_sock.getsockname()[1]

        bluetooth.advertise_service(
            self._server_sock,
            BT_SERVICE_NAME,
            service_id      = SMART_GLASSES_UUID,
            service_classes = [SMART_GLASSES_UUID, bluetooth.SERIAL_PORT_CLASS],
            profiles        = [bluetooth.SERIAL_PORT_PROFILE],
            description     = "Smart Glasses assistive audio — 16 kHz PCM16 mono",
        )
        log.info("SDP service %r advertised on RFCOMM channel %d",
                 BT_SERVICE_NAME, actual_ch)

        self._running = True
        self._accept_thread.start()
        self._send_thread.start()

        try:
            local_addr = bluetooth.read_local_bdaddr()
        except Exception:
            local_addr = "??"

        return (f"Bluetooth RFCOMM  addr={local_addr}  "
                f"ch={actual_ch}  service={BT_SERVICE_NAME!r}")

    def stop(self):
        self._running = False
        try:
            bluetooth.stop_advertising(self._server_sock)
        except Exception:
            pass
        for sock in [self._client_sock, self._server_sock]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

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
            except: pass
            try:    self._send_queue.put_nowait(pcm)
            except: pass

    # ── Internal threads ──────────────────────────────────────────────────────

    def _accept_loop(self):
        """Block on accept(), send WAV header, register the new client."""
        while self._running:
            try:
                self._server_sock.settimeout(1.0)
                conn, info = self._server_sock.accept()
            except bluetooth.btcommon.BluetoothError:
                continue   # timeout — loop and re-check _running
            except OSError:
                break

            addr, ch = info
            log.info("Phone connected via BT: %s  ch%d", addr, ch)
            print(f"\n  ✓  Phone connected via Bluetooth: {addr}")

            # Send WAV header so phone_receiver.py can skip it
            try:
                conn.sendall(_make_wav_header())
            except bluetooth.btcommon.BluetoothError:
                conn.close()
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
            except (bluetooth.btcommon.BluetoothError, OSError):
                log.info("BT link dropped — waiting for phone to reconnect ...")
                print("  ✗  BT link dropped — waiting for phone to reconnect ...")
                with self._client_lock:
                    self._client_sock = None
