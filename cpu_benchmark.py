import argparse
import math
import os
import platform
import shlex
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmark import PORT, post, seconds, wait
from common import distance, normalize
from speech_tests import LANGUAGES, fleurs, parliament, quiet


def wer(items, hyps, lang):
    errors = sum(distance(normalize(r, lang), normalize(h, lang)) for (_, r), h in zip(items, hyps))
    return 100 * errors / sum(len(normalize(r, lang)) for _, r in items)


def memory_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    return int(out.strip() or 0) // 1024


def row(*cells):
    print("\t".join(str(c) for c in cells), flush=True)


def report(name, size, mode, items, parts, hyps, times, load, rss, lang):
    audio = sum(seconds(w) for w, _ in items)
    scores = []
    start = 0
    for part in parts:
        scores.append(f"{wer(part, hyps[start : start + len(part)], lang):.1f}")
        start += len(part)
    ordered = sorted(times)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    row(name, size, mode, *scores, f"{sum(times) / audio:.3f}", f"{1000 * statistics.median(times):.0f}", f"{1000 * p95:.0f}", f"{load:.1f}", rss)


def whisper(args, name, path, modes, items, parts):
    for mode in modes:
        command = [args.server, "-m", path, "-ng", "-t", str(args.threads), "--host", "127.0.0.1", "--port", str(PORT), *shlex.split(args.server_args)]
        begun = time.perf_counter()
        server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait()
            load = time.perf_counter() - begun
            post(items[0][0], language=args.lang, response_format="json", temperature="0")
            hyps, times = [], []
            for wav, _ in items:
                extra = {"audio_ctx": str(min(1500, max(300, math.ceil((seconds(wav) + 1) * 50))))} if mode == "short" else {}
                begun = time.perf_counter()
                hyps.append(post(wav, language=args.lang, response_format="json", temperature="0", no_language_probabilities="true", **extra)["text"])
                times.append(time.perf_counter() - begun)
            rss = memory_mb(server.pid)
            if args.silence and mode == modes[0]:
                texts = [post(w, language=args.lang, response_format="json", temperature="0")["text"].strip()[:60] for w in quiet(Path(args.data) / args.lang, np.random.default_rng(1))]
                row(f"# {name} on silence and noise", texts)
        finally:
            server.terminate()
            server.wait()
        report(name, os.path.getsize(path) // 1_000_000, mode, items, parts, hyps, times, load, rss, args.lang)


def parakeet(args, name, path, items, parts):
    def run(files):
        begun = time.perf_counter()
        subprocess.run([args.parakeet_cli, "-m", path, "-ng", "-t", str(args.threads), "-otxt", *map(str, files)], check=True, capture_output=True)
        return time.perf_counter() - begun

    files = [w for w, _ in items]
    one = run(files[:1])
    total = run(files)
    hyps = []
    for f in files:
        txt = Path(str(f) + ".txt")
        hyps.append(txt.read_text())
        txt.unlink()
    per_clip = (total - one) / (len(files) - 1)
    audio = sum(seconds(w) for w, _ in items[1:])
    scores = []
    start = 0
    for part in parts:
        scores.append(f"{wer(part, hyps[start : start + len(part)], args.lang):.1f}")
        start += len(part)
    row(name, os.path.getsize(path) // 1_000_000, "cli", *scores, f"{(total - one) / audio:.3f}", f"{1000 * per_clip:.0f}", "-", "-", "-")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="name=path/to/ggml.bin")
    p.add_argument("--server", default="whisper-server")
    p.add_argument("--parakeet", help="name=path/to/parakeet.bin")
    p.add_argument("--parakeet-cli", default="parakeet-cli")
    p.add_argument("--lang", choices=LANGUAGES, default="sr")
    p.add_argument("--server-args", default="", help="extra whisper-server flags")
    p.add_argument("--silence", action="store_true", help="also transcribe five clips without speech")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--clips", type=int, default=30)
    p.add_argument("--modes", default="full,short")
    p.add_argument("--data", default="data/speech")
    args = p.parse_args()
    folder, shard = LANGUAGES[args.lang]
    root = Path(args.data) / args.lang
    root.mkdir(parents=True, exist_ok=True)
    parts = [fleurs(root, folder, 400)[: args.clips], parliament(root, args.lang, shard, args.clips)]
    items = parts[0] + parts[1]
    print(f"# {platform.machine()} {platform.system()} threads {args.threads} language {args.lang} clips {len(items)} audio {sum(seconds(w) for w, _ in items):.0f} s", flush=True)
    row("model", "MB", "context", "WER fleurs", "WER parliament", "real-time factor", "median ms", "p95 ms", "load s", "memory MB")
    for spec in args.models:
        name, path = spec.split("=", 1)
        whisper(args, name, path, args.modes.split(","), items, parts)
    if args.parakeet:
        name, path = args.parakeet.split("=", 1)
        parakeet(args, name, path, items, parts)


if __name__ == "__main__":
    main()
