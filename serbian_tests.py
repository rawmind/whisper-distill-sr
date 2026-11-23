import argparse
import io
import math
import random
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmark import PORT, post, seconds, wait
from common import distance, normalize, sample_test
from prepare import BASE, fetch, stream

SHARD = "https://huggingface.co/datasets/classla/ParlaSpeech-RS/resolve/main/data/train-00137-of-00138.parquet"


def fleurs(root, count):
    base = root / "fleurs"
    if not (base / "test.tsv").exists():
        fetch(f"{BASE}/sr_rs/test.tsv", base / "test.tsv")
    refs = sample_test(base / "test.tsv", count)
    (base / "wav").mkdir(exist_ok=True)
    if any(not (base / "wav" / n).exists() for n in refs):
        stream("sr_rs", "test", set(refs), lambda n, a: sf.write(base / "wav" / n, a, 16000, subtype="PCM_16"))
    return [(base / "wav" / n, r) for n, r in refs.items()]


def parliament(root, count):
    import pyarrow.parquet as pq

    base = root / "parliament"
    if not (base / "refs.tsv").exists():
        (base / "wav").mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SHARD, base / "shard.parquet")
        table = pq.read_table(base / "shard.parquet")
        order = list(range(table.num_rows))
        random.Random(5).shuffle(order)
        lines = []
        for i in order:
            row = table.slice(i, 1).to_pylist()[0]
            text = (row["text_normalised"] or "").strip()
            if not 4 <= row["audio_length"] <= 25 or not text:
                continue
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            assert rate == 16000
            name = f"{len(lines):04d}.wav"
            sf.write(base / "wav" / name, audio if audio.ndim == 1 else audio.mean(1), 16000, subtype="PCM_16")
            lines.append(f"{name}\t{text}")
            if len(lines) == 150:
                break
        (base / "refs.tsv").write_text("\n".join(lines))
        (base / "shard.parquet").unlink()
    rows = [line.split("\t") for line in (base / "refs.tsv").read_text().splitlines()]
    return [(base / "wav" / n, t) for n, t in rows[:count]]


def add_noise(root, items, snr, rng):
    out = root / f"noise{snr}"
    out.mkdir(exist_ok=True)
    made = []
    for wav, ref in items:
        audio, _ = sf.read(wav, dtype="float32")
        spectrum = np.fft.rfft(rng.standard_normal(len(audio)))
        noise = np.fft.irfft(spectrum / np.sqrt(np.maximum(1, np.arange(len(spectrum)))), len(audio))
        mixed = audio + noise * np.sqrt((audio**2).mean() / (noise**2).mean() / 10 ** (snr / 10))
        sf.write(out / wav.name, mixed / max(1, np.abs(mixed).max()), 16000, subtype="PCM_16")
        made.append((out / wav.name, ref))
    return made


def quiet(root, rng):
    out = root / "quiet"
    out.mkdir(exist_ok=True)
    made = []
    for i, level in enumerate([None, -50, -50, -20, -20]):
        audio = np.zeros(160000) if level is None else rng.standard_normal(160000) * 10 ** (level / 20)
        sf.write(out / f"{i}.wav", audio.astype("float32"), 16000, subtype="PCM_16")
        made.append(out / f"{i}.wav")
    return made


def score(items, hyps):
    errors = [distance(normalize(r, "sr"), normalize(h, "sr")) for (_, r), h in zip(items, hyps)]
    words = [len(normalize(r, "sr")) for _, r in items]
    rng = random.Random(0)
    samples = []
    for _ in range(1000):
        pick = [rng.randrange(len(errors)) for _ in errors]
        samples.append(100 * sum(errors[i] for i in pick) / sum(words[i] for i in pick))
    samples.sort()
    return f"WER {100 * sum(errors) / sum(words):.1f}\t95% {samples[25]:.1f}-{samples[974]:.1f}\twords {sum(words)}"


def run(items, short):
    hyps = []
    for wav, _ in items:
        extra = {"audio_ctx": str(min(1500, max(300, math.ceil((seconds(wav) + 1) * 50))))} if short else {}
        hyps.append(post(wav, language="sr", response_format="json", temperature="0", no_language_probabilities="true", **extra)["text"])
    return hyps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="name=path/to/ggml.bin")
    p.add_argument("--server", default="whisper-server")
    p.add_argument("--data", default="data/serbian")
    p.add_argument("--clips", type=int, default=400)
    args = p.parse_args()
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    items = fleurs(root, args.clips)
    parl = parliament(root, min(150, args.clips))
    first = items[:100]
    tests = [
        ("fleurs", items, False),
        ("parliament", parl, False),
        ("noise 10 dB", add_noise(root, first, 10, rng), False),
        ("noise 5 dB", add_noise(root, first, 5, rng), False),
        ("fleurs, short context", first, True),
        ("parliament, short context", parl[:100], True),
    ]
    silent = quiet(root, rng)
    for spec in args.models:
        name, path = spec.split("=", 1)
        server = subprocess.Popen([args.server, "-m", path, "--host", "127.0.0.1", "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait()
            for tag, clips, short in tests:
                print(f"{name}\t{tag}\t{len(clips)} clips\t{score(clips, run(clips, short))}", flush=True)
            print(f"{name}\tsilence and noise\t{[post(w, language='sr', response_format='json', temperature='0')['text'].strip()[:60] for w in silent]}", flush=True)
        finally:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
