"""Teleport sound effects, synthesized at startup and played with macOS `afplay`.

Each effect gets a sound matched to its duration; reappearing plays it reversed.
Drop your own file at sounds/<effect>-vanish.<wav|mp3|m4a|aiff> (or -appear) to override.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from .effects import EFFECTS

SR = 44100
SOUNDS_DIR = Path(__file__).resolve().parent.parent / "sounds"
GENERATED_DIR = SOUNDS_DIR / "generated"
CUSTOM_EXTS = (".wav", ".mp3", ".m4a", ".aiff")


def _t(duration: float) -> np.ndarray:
    return np.arange(int(SR * duration)) / SR


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n)
    a, r = int(SR * attack), int(SR * release)
    env[:a] = np.linspace(0, 1, a) ** 2
    env[n - r :] *= np.linspace(1, 0, r) ** 2
    return env


def _band_noise(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-9)


def _chirp(t: np.ndarray, f0: float, f1: float, curve: float = 2.0) -> np.ndarray:
    """Sine sweeping from f0 to f1, easing in with the given curve."""
    frac = (t / t[-1]) ** curve
    freq = f0 + (f1 - f0) * frac
    return np.sin(2 * np.pi * np.cumsum(freq) / SR)


def _zip(rng: np.random.Generator) -> np.ndarray:
    """Anime-style instant-teleport "zip": a fast metallic upward sweep with a ringing tail."""
    t = _t(0.55)
    n = len(t)
    # Pitch shoots up over ~0.1s, overshoots, settles into a ring.
    freq = np.where(
        t < 0.1,
        500 * (3200 / 500) ** (t / 0.1),
        2200 + 1000 * np.exp(-(t - 0.1) * 30),
    )
    freq *= 1 + 0.015 * np.sin(2 * np.pi * 28 * t)  # slight warble
    phase = 2 * np.pi * np.cumsum(freq) / SR
    metallic = np.sin(phase + 1.8 * np.sin(phase * 2.01))  # FM gives the buzzy, metallic edge
    body = 0.7 * metallic + 0.3 * np.sin(phase * 0.5)
    env = np.minimum(t / 0.008, 1) * np.exp(-np.maximum(t - 0.1, 0) * 9)
    air = _band_noise(n, 2500, 9000, rng) * np.exp(-t * 25)
    return body * env + 0.35 * air


def _charge(duration: float, rng: np.random.Generator) -> np.ndarray:
    """Low energy hum that builds up to the jump."""
    t = _t(duration)
    rise = (t / duration) ** 2
    hum = np.sin(2 * np.pi * (110 + 90 * rise) * t) + 0.5 * np.sin(2 * np.pi * (220 + 260 * rise) * t)
    hum *= 0.7 + 0.3 * np.sin(2 * np.pi * (6 + 24 * rise) * t)
    crackle = _band_noise(len(t), 3000, 8000, rng) * rise * 0.25
    return (0.3 * hum + crackle) * rise


def beam(duration: float, rng: np.random.Generator) -> np.ndarray:
    """Charge-up hum that peaks in a zip as the person disappears."""
    n = int(SR * duration)
    sig = np.zeros(n)
    z = _zip(rng)
    zip_at = int(SR * duration * 0.72)  # matches when the beam effect finishes erasing the person
    sig[:zip_at] = _charge(zip_at / SR, rng)[:zip_at]
    end = min(n, zip_at + len(z))
    sig[zip_at:end] += z[: end - zip_at]
    if end - zip_at < len(z):  # let the ring tail spill past the effect
        sig = np.concatenate([sig, z[end - zip_at :]])
    return sig * _envelope(len(sig), 0.3, 0.05)


def beam_appear(duration: float, rng: np.random.Generator) -> np.ndarray:
    """Arrive with the zip right away, then a fading shimmer as the person materializes."""
    n = int(SR * duration)
    t = _t(duration)
    sig = np.zeros(n)
    z = _zip(rng)
    sig[: len(z)] += z[:n]
    settle = np.sin(2 * np.pi * (300 - 120 * t / duration) * t) * np.exp(-t * 2.5) * 0.25
    sig += settle * np.clip(t / 0.1, 0, 1)
    return sig * _envelope(n, 0.004, 0.3)


def dissolve(duration: float, rng: np.random.Generator) -> np.ndarray:
    """Crackling fizz that builds and burns out, over a falling tone."""
    t = _t(duration)
    n = len(t)
    fizz = _band_noise(n, 2500, 10000, rng)
    crackle = np.zeros(n)
    for start in rng.integers(0, n - 2000, int(70 * duration)):
        length = rng.integers(200, 1500)
        crackle[start : start + length] += rng.uniform(0.3, 1) * np.exp(-np.arange(length) / (length / 5))
    crackle *= _band_noise(n, 800, 6000, rng)
    tone = _chirp(t, 600, 150, 1.0) * 0.5
    swell = np.sin(np.pi * np.clip(t / duration * 1.1, 0, 1)) ** 1.5
    sig = (0.3 * fizz + 0.8 * crackle) * swell + 0.25 * tone
    return sig * _envelope(n, 0.05, 0.3)


def particles(duration: float, rng: np.random.Generator) -> np.ndarray:
    """A gust of wind carrying tiny glassy chimes."""
    t = _t(duration)
    n = len(t)
    gust = np.sin(np.pi * t / duration) ** 1.2
    wind = _band_noise(n, 200, 1200, rng) * 0.6 + _band_noise(n, 1200, 4000, rng) * 0.4
    wind *= gust * (0.8 + 0.2 * np.sin(2 * np.pi * 1.7 * t))
    chimes = np.zeros(n)
    ct = _t(0.25)
    for _ in range(int(25 * duration)):
        start = int(rng.uniform(0, 0.75) * n)
        f = rng.choice([1568, 1760, 2093, 2349, 2637, 3136])
        ping = np.sin(2 * np.pi * f * ct) * np.exp(-ct * 22)
        end = min(n, start + len(ct))
        chimes[start:end] += ping[: end - start] * rng.uniform(0.2, 0.6)
    sig = 0.45 * wind + 0.35 * chimes
    return sig * _envelope(n, 0.15, 0.3)


SYNTHS = {"beam": beam, "dissolve": dissolve, "particles": particles}
APPEAR_SYNTHS = {"beam": beam_appear}  # others play their vanish sound reversed


def _write_wav(path: Path, sig: np.ndarray) -> None:
    sig = sig / (np.abs(sig).max() + 1e-9) * 0.8
    pcm = (sig * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def generate_all() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)  # fixed seed: same sounds every run
    for name, synth in SYNTHS.items():
        sig = synth(EFFECTS[name].duration, rng)
        _write_wav(GENERATED_DIR / f"{name}-vanish.wav", sig)
        appear = APPEAR_SYNTHS[name](EFFECTS[name].duration, rng) if name in APPEAR_SYNTHS else sig[::-1]
        _write_wav(GENERATED_DIR / f"{name}-appear.wav", appear)


class SoundPlayer:
    def __init__(self, enabled: bool = True, volume: float = 1.0) -> None:
        self.enabled = enabled
        self.volume = volume
        self._proc: subprocess.Popen | None = None
        if enabled:
            generate_all()

    def path_for(self, effect: str, direction: str) -> Path | None:
        for ext in CUSTOM_EXTS:
            custom = SOUNDS_DIR / f"{effect}-{direction}{ext}"
            if custom.exists():
                return custom
        generated = GENERATED_DIR / f"{effect}-{direction}.wav"
        return generated if generated.exists() else None

    def play(self, effect: str, direction: str) -> None:
        if not self.enabled:
            return
        path = self.path_for(effect, direction)
        if path is None:
            return
        self.stop()
        self._proc = subprocess.Popen(
            ["afplay", "-v", str(self.volume), str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if self.enabled:
            generate_all()
        else:
            self.stop()
        return self.enabled
