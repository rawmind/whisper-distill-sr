import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmark import PORT, wait
from common import normalize
from srt_test import cues, decode

RATE = 16000
CONDITIONS = [("fast", False), ("fast", True), ("backlog", False), ("backlog", True)]


def from_set(folder, gap):
    base = Path(folder)
    rows = [line.split("\t", 1) for line in (base / "refs.tsv").read_text().splitlines() if line]
    silence = np.zeros(int(gap * RATE), dtype=np.float32)
    parts, texts = [], []
    for name, text in sorted(rows):
        audio, rate = sf.read(base / "wav" / name, dtype="float32")
        assert rate == RATE
        parts += [audio if audio.ndim == 1 else audio.mean(1), silence]
        texts.append(text)
    return np.concatenate(parts), " ".join(texts)


def from_srt(media, srt):
    return decode(media), " ".join(text for _, _, text in cues(srt))


def edits(ref, hyp):
    rows, cols = len(ref) + 1, len(hyp) + 1
    prev = [(j, 0, j, 0) for j in range(cols)]
    for i in range(1, rows):
        cur = [(i, 0, 0, i)]
        for j in range(1, cols):
            same = ref[i - 1] == hyp[j - 1]
            d, s, n, x = prev[j - 1]
            options = [(d + (not same), s + (not same), n, x)]
            d, s, n, x = cur[j - 1]
            options.append((d + 1, s, n + 1, x))
            d, s, n, x = prev[j]
            options.append((d + 1, s, n, x + 1))
            cur.append(min(options))
        prev = cur
    total, subs, ins, dels = prev[-1]
    return {"errors": total, "subs": subs, "ins": ins, "dels": dels, "words": len(ref)}


def run(runner, titl, audio_file, lang, mode, short):
    out = subprocess.run(
        ["node", str(runner), "--titl", titl, "--server", f"http://127.0.0.1:{PORT}/inference", "--audio", audio_file, "--language", lang, "--mode", mode, *(["--short"] if short else [])],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def main():
    p = argparse.ArgumentParser(description="run Titl's lookahead pipeline on long audio and score the stitched text")
    p.add_argument("models", nargs="+", help="name=path/to/ggml.bin")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--set", help="folder with wav/ and refs.tsv, joined in order")
    src.add_argument("--media", help="a recording; needs --srt")
    p.add_argument("--srt")
    p.add_argument("--gap", type=float, default=0.5, help="silence between clips of --set, seconds")
    p.add_argument("--out", required=True, help="folder for pipeline.txt and pipeline.json")
    p.add_argument("--lang", default="sr")
    p.add_argument("--titl", default=str(Path.home() / "projects/titl"))
    p.add_argument("--server", default="whisper-server")
    p.add_argument("--vad-model", help="Silero model; adds --vad like Titl")
    args = p.parse_args()
    audio, ref = from_set(args.set, args.gap) if args.set else from_srt(args.media, args.srt)
    ref_words = normalize(ref, args.lang)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("pipeline_runner.mjs")
    vad = ["--vad", "--vad-model", args.vad_model] if args.vad_model else []
    results = json.loads((out / "pipeline.json").read_text()) if (out / "pipeline.json").exists() else {}
    report = open(out / "pipeline.txt", "a")
    with tempfile.NamedTemporaryFile(suffix=".f32") as tmp:
        tmp.write(audio.astype(np.float32).tobytes())
        tmp.flush()
        for spec in args.models:
            name, path = spec.split("=", 1)
            server = subprocess.Popen([args.server, "-m", path, "--host", "127.0.0.1", "--port", str(PORT), *vad], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                wait()
                for mode, short in CONDITIONS:
                    tag = f"{mode}, {'short' if short else 'full'} context"
                    got = run(runner, args.titl, tmp.name, args.lang, mode, short)
                    score = edits(ref_words, normalize(got["hyp"], args.lang))
                    sizes = [w["seconds"] for w in got["windows"]]
                    line = (
                        f"{name}\t{tag}\t{len(audio) / RATE / 60:.1f} min\tWER {100 * score['errors'] / score['words']:.1f}"
                        f"\tsub {score['subs']} ins {score['ins']} del {score['dels']} of {score['words']}"
                        f"\twindows {len(sizes)} median {sorted(sizes)[len(sizes) // 2] if sizes else 0:.1f} s\terrors {len(got['errors'])}"
                    )
                    print(line, flush=True)
                    report.write(line + "\n")
                    report.flush()
                    results.setdefault(name, {})[tag] = {**score, "hyp": got["hyp"], "cues": got["cues"], "windows": got["windows"], "request_errors": got["errors"]}
                    (out / "pipeline.json").write_text(json.dumps(results, ensure_ascii=False))
            finally:
                server.terminate()
                server.wait()


if __name__ == "__main__":
    main()
