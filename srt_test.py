import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000


def seconds(stamp):
    h, m, rest = stamp.strip().split(":")
    s, ms = rest.replace(".", ",").split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def cues(path):
    out = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8-sig").strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        start, end = lines[timing].split("-->")
        text = " ".join(re.sub(r"<[^>]+>", "", line) for line in lines[timing + 1 :]).strip()
        if text:
            out.append((seconds(start), seconds(end.split()[0]), text))
    return out


def groups(items, shortest, longest):
    out, current = [], []
    for start, end, text in items:
        if current and (end - current[0][0] > longest or start - current[-1][1] > 1.0):
            out.append(current)
            current = []
        current.append((start, end, text))
        if current[-1][1] - current[0][0] >= shortest:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return [g for g in out if shortest <= g[-1][1] - g[0][0] <= longest]


def decode(media):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="cut a recording and its SRT into a test set for speech_tests.py")
    p.add_argument("media")
    p.add_argument("srt")
    p.add_argument("--out", required=True, help="e.g. data/speech/sr/stories")
    p.add_argument("--min", type=float, default=4)
    p.add_argument("--max", type=float, default=20)
    args = p.parse_args()
    audio = decode(args.media)
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    lines = []
    for i, group in enumerate(groups(cues(args.srt), args.min, args.max)):
        start, end = group[0][0], group[-1][1]
        name = f"{i:04d}.wav"
        sf.write(out / "wav" / name, audio[int(start * RATE) : int(end * RATE)], RATE, subtype="PCM_16")
        lines.append(f"{name}\t{' '.join(t for _, _, t in group)}")
    (out / "refs.tsv").write_text("\n".join(lines) + "\n")
    total = sum(sf.info(out / "wav" / line.split("\t")[0]).frames for line in lines) / RATE
    print(f"{len(lines)} clips, {total / 60:.1f} min of {len(audio) / RATE / 60:.1f} min, written to {out}")


if __name__ == "__main__":
    main()
