import argparse
import shlex
import statistics
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmark import PORT, post, wait
from speech_tests import LANGUAGES, parliament

RATE = 16000


def trimmed(path):
    audio, _ = sf.read(path, dtype="float32")
    frame = RATE // 50
    rms = np.sqrt((audio[: len(audio) // frame * frame].reshape(-1, frame) ** 2).mean(1))
    loud = np.flatnonzero(rms > max(0.004, rms.max() * 0.05))
    return audio[loud[0] * frame : (loud[-1] + 1) * frame]


def build(root, items, rng, count, kind):
    out = root / f"timing_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    clips = [a for a in (trimmed(w) for w, _ in items) if 3 * RATE <= len(a) <= (5 if kind == "window" else 9) * RATE]
    made = []
    for i in range(count):
        parts = [clips[rng.integers(len(clips))]] if kind == "window" else [clips[j] for j in rng.choice(len(clips), 2, replace=False)]
        pieces, speech, at = [], [], 0.0

        def silence(seconds):
            nonlocal at
            pieces.append(rng.standard_normal(int(seconds * RATE)).astype("float32") * 0.001)
            at += int(seconds * RATE) / RATE

        silence(rng.uniform(1, 3))
        for n, part in enumerate(parts):
            if n:
                silence(rng.uniform(1.5, 4))
            speech.append((at, at + len(part) / RATE))
            pieces.append(part)
            at += len(part) / RATE
        silence(8 - at if kind == "window" and at < 8 else rng.uniform(1, 2))
        path = out / f"{i:03d}.wav"
        sf.write(path, np.concatenate(pieces), RATE, subtype="PCM_16")
        made.append((path, speech, at))
    return made


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def measure(made, lang):
    starts, ends, silent, missed, silence_total, speech_total, empty, odd = [], [], 0.0, 0.0, 0.0, 0.0, 0, 0
    for wav, speech, length in made:
        segments = [(s["start"], s["end"]) for s in post(wav, language=lang, response_format="verbose_json", temperature="0", no_language_probabilities="true").get("segments", []) if s["text"].strip()]
        empty += not segments
        odd += any(e <= b or e > length + 1 for b, e in segments)
        covered = sum(overlap(seg, sp) for seg in segments for sp in speech)
        shown = sum(e - b for b, e in segments)
        silent += max(0.0, shown - covered)
        spoken = sum(e - b for b, e in speech)
        missed += max(0.0, spoken - covered)
        speech_total += spoken
        silence_total += length - spoken
        for begin, end in speech:
            if segments:
                starts.append(min((b - begin for b, _ in segments), key=abs))
                ends.append(min((e - end for _, e in segments), key=abs))
    late = [abs(x) for x in starts]
    return {
        "start error median s": round(statistics.median(late), 2),
        "p90": round(percentile(late, 0.9), 2),
        "starts within 0.5 s": f"{100 * sum(x <= 0.5 for x in late) / len(late):.0f}%",
        "median signed start": round(statistics.median(starts), 2),
        "end error median s": round(statistics.median(abs(x) for x in ends), 2),
        "ends within 0.5 s": f"{100 * sum(abs(x) <= 0.5 for x in ends) / len(ends):.0f}%",
        "subtitle over silence": f"{100 * silent / silence_total:.0f}%",
        "speech without subtitle": f"{100 * missed / speech_total:.0f}%",
        "clips without text": empty,
        "clips with odd times": odd,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="name=path/to/ggml.bin")
    p.add_argument("--server", default="whisper-server")
    p.add_argument("--vad-model", required=True)
    p.add_argument("--lang", choices=LANGUAGES, default="sr")
    p.add_argument("--clips", type=int, default=60)
    p.add_argument("--data", default="data/speech")
    args = p.parse_args()
    root = Path(args.data) / args.lang
    items = parliament(root, args.lang, LANGUAGES[args.lang][1], 150)
    sets = {kind: build(root, items, np.random.default_rng(3), args.clips, kind) for kind in ("pair", "window")}
    for spec in args.models:
        name, path = spec.split("=", 1)
        for label, extra in (("no VAD", ""), ("VAD", f"--vad --vad-model {shlex.quote(args.vad_model)}")):
            command = [args.server, "-m", path, "--host", "127.0.0.1", "--port", str(PORT), *shlex.split(extra)]
            server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                wait()
                for kind, made in sets.items():
                    print(f"{name}\t{label}\t{kind}\t{len(made)} clips\t" + "\t".join(f"{k} {v}" for k, v in measure(made, args.lang).items()), flush=True)
            finally:
                server.terminate()
                server.wait()


if __name__ == "__main__":
    main()
