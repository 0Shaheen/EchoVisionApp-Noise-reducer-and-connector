"""
audio_streamer.py — TCP Audio Stream Server
============================================
Opens a TCP server. The phone connects and receives:
  [44 bytes]  WAV header  (tells the client: 16 kHz, mono, PCM16)
  [∞ bytes]   Raw int16 PCM audio, continuous stream

Compatible with any app that can open a network audio stream:
  VLC (Android/iOS):  Open Network Stream → tcp://rpi_ip:8888
  Android AudioTrack: connect socket, skip 44 bytes, read int16 PCM
  iOS AVAudioEngine:  connect socket, skip 44 bytes, feed PCM buffer
  Any STT SDK:        feed the raw PCM bytes directly as microphone input
"""

import io
import logging
import queue
import socket
import struct
import threading
import wave
import numpy as np

from config import STREAM_HOST, STREAM_PORT, OUT_SAMPLE_RATE, OUT_CHANNELS, OUT_BITS

log = logging.getLogger(__name__)


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _make_stream_wav_header() -> bytes:
    """
    Standard 44-byte WAV header with data size set to 0xFFFFFFFF
    (unknown / streaming). Any audio player or SDK reads this and
    immediately knows the sample rate, bit depth, and channel count.
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


class AudioStreamServer:
    """
    TCP server that streams processed mono PCM16 audio to one connected client.
    Phone can disconnect and reconnect freely — the server always waits.
    """

    def __init__(self):
        self._server_sock   = None
        self._client_sock   = None
        self._client_lock   = threading.Lock()
        self._send_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._running       = False
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="accept")
        self._send_thread   = threading.Thread(
            target=self._send_loop, daemon=True, name="send")

    def start(self) -> str:
        """Start the server. Returns the stream URL to display."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((STREAM_HOST, STREAM_PORT))
        self._server_sock.listen(1)
        self._running = True
        self._accept_thread.start()
        self._send_thread.start()

        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            local_ip = probe.getsockname()[0]
            probe.close()
        except Exception:
            local_ip = "0.0.0.0"

        return f"tcp://{local_ip}:{STREAM_PORT}"

    def stop(self):
        self._running = False
        for s in [self._client_sock, self._server_sock]:
            if s:
                try: s.close()
                except Exception: pass

    def push(self, audio: np.ndarray):
        """Push a float32 chunk to the stream. Drops oldest if buffer full."""
        pcm = _float32_to_pcm16(audio)
        try:
            self._send_queue.put_nowait(pcm)
        except queue.Full:
            try:    self._send_queue.get_nowait()
            except: pass
            try:    self._send_queue.put_nowait(pcm)
            except: pass

    def _accept_loop(self):
        while self._running:
            try:
                self._server_sock.settimeout(1.0)
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            log.info("Phone connected: %s:%d", addr[0], addr[1])
            print(f"\n  ✓  Phone connected: {addr[0]}:{addr[1]}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            try:
                conn.sendall(_make_stream_wav_header())
            except OSError:
                conn.close()
                continue

            with self._client_lock:
                if self._client_sock:
                    try: self._client_sock.close()
                    except: pass
                self._client_sock = conn

    def _send_loop(self):
        while self._running:
            try:
                pcm = self._send_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            with self._client_lock:
                client = self._client_sock
            if client is None:
                continue
            try:
                client.sendall(pcm)
            except (BrokenPipeError, ConnectionResetError, OSError):
                log.info("Phone disconnected — waiting for reconnect ...")
                print("  ✗  Phone disconnected — waiting for reconnect ...")
                with self._client_lock:
                    self._client_sock = None