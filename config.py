# config.py — Smart Glasses RPi Zero 2W Configuration
# =====================================================
# Values marked "<-- CHANGE IF NEEDED" are the ones you may need to adjust.

# ── Audio Capture ────────────────────────────────────────────────────────────────
SAMPLE_RATE         = 16000       # Hz
CHANNELS            = 2           # stereo: Left mic = ch0, Right mic = ch1
CAPTURE_DTYPE       = "int32"     # INMP441: 24-bit audio in 32-bit I2S frames
INMP441_BIT_SHIFT   = 8           # right-shift to recover 24-bit value
CHUNK_FRAMES        = 512         # samples per processing block (~32 ms)
# sounddevice/PortAudio device: an integer index OR a (case-insensitive)
# substring of the device name from `python3 audio_capture.py`.
# NOTE: ALSA-style "hw:card,dev" strings do NOT work here — use the PortAudio
# name. "sysdefault" routes through ALSA (handles format/rate) and works on the
# Google voiceHAT / INMP441 I2S setup.  Use "default" or an index if needed.
ALSA_DEVICE         = "sysdefault"   # <-- CHANGE IF NEEDED (list: python3 audio_capture.py)

# ── Microphone Array ─────────────────────────────────────────────────────────────
MIC_SEPARATION_M    = 0.065       # metres — measure your actual glasses frame
SPEED_OF_SOUND_MS   = 343.0

# ── SRP-PHAT / DOA ───────────────────────────────────────────────────────────────
MAX_DELAY_SAMPLES   = int((MIC_SEPARATION_M / SPEED_OF_SOUND_MS) * SAMPLE_RATE)
FFT_SIZE            = 2048
DOA_WINDOW_SEC      = 0.3
DOA_SMOOTHING       = 0.5
BEAMFORMER_CONFIDENCE_THRESHOLD = 2.0

# ── Beamformer ───────────────────────────────────────────────────────────────────
BEAM_GAIN           = 2.0

# ── Noise Reducer Selection ──────────────────────────────────────────────────────
# False → noise_reducer.py          (original Wiener filter — stable, lighter)
# True  → noise_reducer_improved.py (MMSE-LSA + decision-directed SNR + speech
#         presence probability — more natural speech, better fricative
#         preservation, slightly more CPU)
# Both expose the same WienerFilter class, so only this flag changes which is used.
USE_IMPROVED_NOISE_REDUCER = False

# ── Wiener Filter Noise Reduction ────────────────────────────────────────────────
# Shared by both reducers (the improved one ignores WIENER_OVERESTIMATE).
NOISE_REDUCE_ENABLED  = True
NOISE_FLOOR_DECAY     = 0.998
WIENER_OVERESTIMATE   = 1.5
MAX_REDUCTION_DB      = 12.0
SPECTRAL_FLOOR        = 0.25
CALIB_SEC             = 2.0       # seconds of silence recorded at startup

# ── Bluetooth RFCOMM Stream ──────────────────────────────────────────────────────
# The RPi listens on an RFCOMM channel and (in BlueZ compat mode) publishes a
# Serial Port Profile SDP record.  The EchoVision phone app pairs with the RPi
# once via bluetoothctl, then connects over SPP and receives newline-delimited
# base64 frames of continuous PCM16 mono audio.
BT_SERVICE_NAME     = "SmartGlassesAudio"
BT_RFCOMM_CHANNEL   = 1           # 1–30; auto-selected if the channel is busy

# ── Output Audio Format ──────────────────────────────────────────────────────────
OUT_SAMPLE_RATE     = 16000       # Hz
OUT_CHANNELS        = 1           # mono
OUT_BITS            = 16          # PCM16

# ── Logging ──────────────────────────────────────────────────────────────────────
LOG_LEVEL           = "INFO"
