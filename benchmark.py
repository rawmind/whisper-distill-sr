import argparse
import collections
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from common import LANGS, distance, normalize, sample_test

PORT = 18765


def seconds(wav):
    return (wav.stat().st_size - 44) / 32000


def post(wav, **fields):
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    body += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    body += wav.read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/inference", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.load(res)


def wait():
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
            return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("whisper-server did not start")


def accuracy(name, lang, items):
    code = LANGS[lang]
    post(items[0][0], language=code, response_format="json", temperature="0")
    start = time.perf_counter()
    hyps = [post(w, language=code, response_format="json", temperature="0", no_language_probabilities="true")["text"] for w, _ in items]
    sec = time.perf_counter() - start
    errors = sum(distance(normalize(r, code), normalize(h, code)) for (_, r), h in zip(items, hyps))
    words = sum(len(normalize(r, code)) for _, r in items)
    audio = sum(seconds(w) for w, _ in items)
    print(f"{name}\t{lang}\tWER {100 * errors / words:.1f}\twords {words}\t{audio / sec:.0f}x\t{1000 * sec / len(items):.0f} ms/clip", flush=True)
    return hyps


def timestamps_and_detection(name, lang, items):
    found = collections.Counter()
    odd = 0
    for wav, _ in items[:50]:
        found[str(post(wav, language="auto", response_format="verbose_json", temperature="0").get("language"))] += 1
        segments = post(wav, language=LANGS[lang], response_format="verbose_json", temperature="0").get("segments", [])
        odd += not segments or any(s["end"] <= s["start"] or s["end"] > seconds(wav) + 1.5 for s in segments)
    print(f"{name}\t{lang}\tdetected {found.most_common(4)}\tclips with odd timestamps {odd}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="name=path/to/ggml.bin")
    p.add_argument("--server", default="whisper-server")
    p.add_argument("--data", default="data/test")
    p.add_argument("--clips", type=int, default=100)
    p.add_argument("--langs", default=",".join(LANGS))
    p.add_argument("--checks", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()
    data = {}
    for lang in args.langs.split(","):
        base = Path(args.data) / lang
        refs = sample_test(base / "test.tsv", args.clips)
        data[lang] = [(base / "wav" / n, r) for n, r in refs.items() if (base / "wav" / n).exists()]
    dump = {"refs": {lang: [r for _, r in items] for lang, items in data.items()}}
    for spec in args.models:
        name, path = spec.split("=", 1)
        server = subprocess.Popen([args.server, "-m", path, "--host", "127.0.0.1", "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait()
            for lang, items in data.items():
                dump[f"{name}|{lang}"] = accuracy(name, lang, items)
                if args.checks:
                    timestamps_and_detection(name, lang, items)
        finally:
            server.terminate()
            server.wait()
    if args.out:
        Path(args.out).write_text(json.dumps(dump, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
