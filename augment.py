import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
MAX_SECONDS = 30
NOISE_KINDS = ("white", "brown", "babble", "hum")
NOISE_WEIGHTS = (0.25, 0.25, 0.4, 0.1)


def colored(n, rng, power):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spectrum), dtype=np.float64)
    f[0] = 1
    return np.fft.irfft(spectrum / f ** (power / 2), n)


def babble(n, rng, other):
    out = np.zeros(n)
    for _ in range(rng.integers(3, 7)):
        clip = other()
        if not len(clip):
            continue
        if len(clip) < n:
            clip = np.tile(clip, n // len(clip) + 1)
        start = rng.integers(0, len(clip) - n + 1)
        part = clip[start : start + n]
        out += part / (np.sqrt((part**2).mean()) + 1e-8)
    return out


def hum(n, rng):
    t = np.arange(n) / RATE
    base = rng.choice([50, 60])
    tone = sum(np.sin(2 * np.pi * base * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 6))
    return tone + 0.05 * rng.standard_normal(n)


def noise(kind, n, rng, other):
    if kind == "white":
        return rng.standard_normal(n)
    if kind == "brown":
        return colored(n, rng, 2)
    if kind == "babble":
        return babble(n, rng, other)
    return hum(n, rng)


def mix(audio, extra, snr):
    signal = (audio**2).mean()
    power = (extra**2).mean()
    if signal == 0 or power == 0:
        return audio
    return audio + extra * np.sqrt(signal / power / 10 ** (snr / 10))


def convolve(audio, kernel):
    size = 1 << (len(audio) + len(kernel) - 2).bit_length()
    return np.fft.irfft(np.fft.rfft(audio, size) * np.fft.rfft(kernel, size), size)[: len(audio)]


def reverb(audio, rng):
    rt60 = rng.uniform(0.2, 0.8)
    t = np.arange(int(rt60 * RATE)) / RATE
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    tail[0] = 0
    kernel = tail / np.sqrt((tail**2).sum()) * rng.uniform(0.3, 1.0)
    kernel[0] = 1
    wet = convolve(audio, kernel)
    return wet * np.sqrt((audio**2).mean() / max((wet**2).mean(), 1e-12))


def lowpass(audio, cutoff):
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / RATE)
    spectrum *= np.clip((cutoff * 1.15 - freqs) / (cutoff * 0.15), 0, 1)
    return np.fft.irfft(spectrum, len(audio))


def mulaw(audio, bits=8):
    mu = 2**bits - 1
    peak = max(np.abs(audio).max(), 1e-8)
    x = audio / peak
    coded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    coded = np.round((coded + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(coded) * ((1 + mu) ** np.abs(coded) - 1) / mu * peak


def speed(audio, factor):
    positions = np.arange(0, len(audio) - 1, factor)
    return np.interp(positions, np.arange(len(audio)), audio)


def augment(audio, rng, other, clean_share=0.2):
    audio = np.asarray(audio, dtype=np.float64)
    applied = []
    if rng.random() < clean_share or not len(audio):
        return audio.astype(np.float32), applied
    if rng.random() < 0.3:
        slowest = max(0.9, len(audio) / (MAX_SECONDS * RATE))
        factor = rng.uniform(slowest, 1.1) if slowest < 1.1 else 1.0
        audio = speed(audio, factor)
        applied.append(f"speed {factor:.2f}")
    if rng.random() < 0.3:
        audio = reverb(audio, rng)
        applied.append("reverb")
    if rng.random() < 0.8:
        kind = rng.choice(NOISE_KINDS, p=NOISE_WEIGHTS)
        snr = rng.uniform(0, 20)
        audio = mix(audio, noise(kind, len(audio), rng, other), snr)
        applied.append(f"{kind} {snr:.0f} dB")
    if rng.random() < 0.25:
        cutoff = rng.uniform(3400, 7000)
        audio = lowpass(audio, cutoff)
        applied.append(f"lowpass {cutoff:.0f} Hz")
    if rng.random() < 0.1:
        audio = mulaw(audio)
        applied.append("mu-law")
    audio *= 10 ** (rng.uniform(-6, 6) / 20)
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio *= 0.99 / peak
    return audio[: MAX_SECONDS * RATE].astype(np.float32), applied


def main():
    p = argparse.ArgumentParser(description="write augmented examples of a few training clips to listen to")
    p.add_argument("wavs", nargs="+")
    p.add_argument("--out", default="work/aug_samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    clips = [sf.read(w, dtype="float32")[0] for w in args.wavs]
    clips = [c.mean(1) if c.ndim > 1 else c for c in clips]
    other = lambda: clips[rng.integers(len(clips))]
    for i, clip in enumerate(clips):
        audio, applied = augment(clip, rng, other, clean_share=0)
        name = out / f"{i:02d}.wav"
        sf.write(name, audio, RATE, subtype="PCM_16")
        print(name, ", ".join(applied) or "clean", flush=True)


if __name__ == "__main__":
    main()
