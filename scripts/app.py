import io
import random
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import streamlit as st

# -----------------------------
# Config
# -----------------------------
ASSETS_DIR = Path("assets/smurfs")
SAMPLE_RATE = 44_100

SMURFISH_NAMES = [
    "Blue Buddy", "Papa-ish", "Smurfette-ish", "Brainy-ish",
    "Hefty-ish", "Jokey-ish", "Clumsy-ish", "Harmony-ish", "Chef-ish",
    "Handy-ish", "Painter-ish", "Poet-ish", "Grouchy-ish"
]

# -----------------------------
# Helpers: Graphics
# -----------------------------
def generate_smurf_avatar(seed: int = None, size: int = 512) -> Image.Image:
    """
    Generate a Smurf-inspired (original) avatar: blue face + white hat.
    No external assets required.
    """
    rnd = random.Random(seed)
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)

    # Background (soft gradient-like bands)
    for y in range(size):
        tone = 245 - int(10 * (y / size))
        d.line([(0, y), (size, y)], fill=(tone, tone, 255, 255))

    # Face (blue circle)
    cx, cy = size // 2, int(size * 0.58)
    radius = int(size * 0.3)
    face_color = (rnd.randint(40, 70), rnd.randint(130, 180), 255, 255)
    d.ellipse(
        [(cx - radius, cy - radius), (cx + radius, cy + radius)],
        fill=face_color,
        outline=(0, 80, 160, 255),
        width=4,
    )

    # Hat (white, stylized cap)
    hat_height = int(size * 0.38)
    hat_width = int(radius * 2.2)
    hat_top = cy - radius - hat_height + int(size * 0.04)
    hat_left = cx - hat_width // 2
    hat_right = cx + hat_width // 2
    # Main hat
    d.pieslice(
        [(hat_left, hat_top), (hat_right, cy - radius + int(size * 0.1))],
        start=180, end=360, fill=(255, 255, 255, 255),
        outline=(180, 180, 180, 255), width=4
    )
    # Hat brim
    brim_h = int(size * 0.06)
    d.rounded_rectangle(
        [(cx - int(hat_width * 0.55), cy - radius - brim_h // 2),
         (cx + int(hat_width * 0.55), cy - radius + brim_h // 2)],
        radius=brim_h // 2,
        fill=(255, 255, 255, 255),
        outline=(180, 180, 180, 255), width=3
    )

    # Eyes
    eye_r = int(radius * 0.16)
    ex_offset = int(radius * 0.38)
    ey = cy - int(radius * 0.15)
    for ex in (cx - ex_offset, cx + ex_offset):
        d.ellipse([(ex - eye_r, ey - eye_r), (ex + eye_r, ey + eye_r)], fill="white", outline=(50, 50, 50, 255), width=3)
        pupil_r = int(eye_r * 0.45)
        d.ellipse([(ex - pupil_r, ey - pupil_r), (ex + pupil_r, ey + pupil_r)], fill=(20, 50, 90, 255))

    # Mouth (open, singing)
    mouth_w = int(radius * 0.9)
    mouth_h = int(radius * 0.55)
    mx0, my0 = cx - mouth_w // 2, cy + int(radius * 0.15)
    mx1, my1 = cx + mouth_w // 2, my0 + mouth_h
    d.ellipse([(mx0, my0), (mx1, my1)], fill=(30, 25, 25, 255), outline=(10, 10, 10, 255), width=4)

    # Tongue
    tongue_h = int(mouth_h * 0.45)
    d.pieslice([(mx0 + 10, my1 - tongue_h), (mx1 - 10, my1 + 20)], start=0, end=180, fill=(230, 100, 120, 255))

    # Musical notes (simple black note glyphs)
    note_count = rnd.randint(2, 4)
    for i in range(note_count):
        nx = rnd.randint(int(size * 0.75), int(size * 0.92))
        ny = rnd.randint(int(size * 0.10), int(size * 0.35))
        # Stem
        d.line([(nx, ny), (nx, ny + 40)], fill=(0, 0, 0, 200), width=4)
        # Head
        d.ellipse([(nx - 10, ny + 28), (nx + 10, ny + 48)], fill=(0, 0, 0, 200))
        # Flag
        d.arc([(nx - 4, ny - 4), (nx + 26, ny + 20)], start=280, end=20, fill=(0, 0, 0, 200), width=4)

    return img

def image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()

def pick_random_image_or_generate(seed: int = None) -> bytes:
    """If licensed assets exist, use them; else generate an avatar."""
    files = []
    if ASSETS_DIR.exists():
        files = [p for p in ASSETS_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if files:
        choice = random.choice(files)
        return Path(choice).read_bytes()
    else:
        img = generate_smurf_avatar(seed=seed)
        return image_to_bytes(img)

# -----------------------------
# Helpers: Audio
# -----------------------------
def sine_wave(frequency: float, duration: float, volume: float = 0.25, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)

    # Simple ADSR-like envelope to avoid clicks
    attack = int(0.03 * len(wave))
    release = int(0.06 * len(wave))
    sustain = len(wave) - attack - release
    if sustain < 0:
        sustain = 0
        attack = max(1, int(0.5 * len(wave)))
        release = len(wave) - attack

    env = np.concatenate([
        np.linspace(0, 1, attack, endpoint=False),
        np.ones(sustain),
        np.linspace(1, 0, release, endpoint=False)
    ])
    wave = wave[:len(env)] * env
    return (volume * wave).astype(np.float32)

def generate_random_melody(seed: int = None, notes: int = 8, sr: int = SAMPLE_RATE) -> bytes:
    rnd = random.Random(seed)
    # C major-ish scale across an octave (plus next C)
    scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    # Build melody
    pieces = []
    for _ in range(notes):
        f = rnd.choice(scale) * (rnd.choice([0.5, 1.0, 1.0, 1.0, 2.0]))  # some variation
        dur = rnd.uniform(0.22, 0.48)
        vol = rnd.uniform(0.18, 0.32)
        pieces.append(sine_wave(f, dur, volume=vol, sr=sr))
        # small rest between notes
        rest = np.zeros(int(sr * rnd.uniform(0.02, 0.07)), dtype=np.float32)
        pieces.append(rest)

    audio = np.concatenate(pieces) if pieces else np.zeros(sr // 2, dtype=np.float32)

    # Normalize to int16 range
    max_abs = np.max(np.abs(audio)) or 1.0
    audio_i16 = (audio / max_abs * 32767).astype(np.int16)

    # Write to WAV in-memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    buf.seek(0)
    return buf.read()

def random_lyrics(seed: int = None, length: int = 24) -> str:
    rnd = random.Random(seed)
    syllables = ["La", "Na", "Da", "Tra", "Laa", "La", "La", "Na"]
    parts = [rnd.choice(syllables) for _ in range(length)]
    return " ".join(parts) + " 🎵"

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Random Smurfs that Sing", page_icon="🎶", layout="centered")

st.title("🎶 Random Smurfs that Sing")
st.caption("Click the button to meet a random (Smurf-inspired) singer and hear a short melody.")

with st.sidebar:
    st.header("Settings")
    seed = st.number_input("Random seed (optional)", value=0, min_value=0, step=1)
    note_count = st.slider("Number of notes", 4, 16, 8)
    show_lyrics = st.checkbox("Show lyrics", value=True)
    st.markdown("---")
    st.write("**Assets**")
    st.write(f"Local images folder: `{ASSETS_DIR.as_posix()}`")
    if ASSETS_DIR.exists():
        count = len([p for p in ASSETS_DIR.glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        st.write(f"Found {count} image(s).")
    else:
        st.write("Folder not found. Using generated avatars.")

if "invocation" not in st.session_state:
    st.session_state.invocation = 0

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Sing a song 🎵", use_container_width=True):
        st.session_state.invocation += 1

inv = st.session_state.invocation
rng = random.Random(None if seed == 0 else seed + inv)

# Pick name & image
name = rng.choice(SMURFISH_NAMES) + f" #{rng.randint(100, 999)}"
image_bytes = pick_random_image_or_generate(seed=rng.randint(0, 1_000_000))

with col1:
    st.subheader(name)
    st.image(image_bytes, caption="Smurf-inspired singer", use_container_width=True)

# Generate audio melody
audio_bytes = generate_random_melody(seed=rng.randint(0, 1_000_000), notes=note_count)

with col2:
    st.subheader("Melody")
    st.audio(audio_bytes, format="audio/wav")

    if show_lyrics:
        st.markdown("### Lyrics")
        st.write(random_lyrics(seed=rng.randint(0, 1_000_000), length=18))

st.markdown("---")
st.caption("Tip: Add your own licensed images to `assets/smurfs/` to customize the look. "
           "This app generates original blue avatars if no images are provided.")
