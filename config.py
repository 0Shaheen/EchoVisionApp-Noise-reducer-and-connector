# config.py — Smart Glasses RPi Zero 2W Configuration
# =====================================================
# Values marked "<-- CHANGE IF NEEDED" are the ones you may need to adjust.

# ── Audio Capture ────────────────────────────────────────────────────────────────
SAMPLE_RATE         = 16000       # Hz
CHANNELS            = 2           # stereo: Left mic = ch0, Right mic = ch1
CAPTURE_DTYPE       = "int32"     # INMP441: 24-bit audio in 32-bit I2S frames
INMP441_BIT_SHIFT   = 8           # right-shift to recover 24-bit value
CHUNK_FRAMES        = 512         # samples per processing block (~32 ms)
ALSA_DEVICE         = "hw:sndrpigooglevoi,0"   # <-- CHANGE IF NEEDED (check: arecord -l)

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

# ── Wiener Filter Noise Reduction ────────────────────────────────────────────────
NOISE_REDUCE_ENABLED  = True
NOISE_FLOOR_DECAY     = 0.998
WIENER_OVERESTIMATE   = 1.5
MAX_REDUCTION_DB      = 12.0
SPECTRAL_FLOOR        = 0.25
CALIB_SEC             = 2.0       # seconds of silence recorded at startup

# ── Bluetooth RFCOMM Stream ──────────────────────────────────────────────────────
# The RPi registers an RFCOMM serial service via SDP.  The EchoVision phone
# app pairs with the RPi once via bluetoothctl, then connects by the SDP
# service name and receives a WAV header followed by continuous PCM16 mono
# audio.  Device class 0x200448 (Hearing Aid) is set in /etc/bluetooth/main.conf
# so Android lists the RPi under Accessibility → Hearing Aids.
BT_SERVICE_NAME     = "SmartGlassesAudio"
BT_RFCOMM_CHANNEL   = 1           # 1–30; BlueZ assigns automatically if in use

# ── Output Audio Format ──────────────────────────────────────────────────────────
OUT_SAMPLE_RATE     = 16000       # Hz
OUT_CHANNELS        = 1           # mono
OUT_BITS            = 16          # PCM16

# ── Logging ──────────────────────────────────────────────────────────────────────
LOG_LEVEL           = "INFO"
