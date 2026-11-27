import argparse
import json
import subprocess
from pathlib import Path

from benchmark import PORT, wait
from common import distance, normalize
from speech_tests import run, score


def main():
    p = argparse.ArgumentParser(description="score models on any folder with wav/ and refs.tsv; results go only to --out")
    p.add_argument("models", nargs="+", help="name=path/to/ggml.bin")
    p.add_argument("--set", required=True, help="folder with wav/ and refs.tsv")
    p.add_argument("--out", required=True, help="folder for results.txt and results.json")
    p.add_argument("--lang", default="sr")
    p.add_argument("--server", default="whisper-server")
    p.add_argument("--server-args", default="")
    args = p.parse_args()
    base = Path(args.set)
    items = [(base / "wav" / n, t) for n, t in (line.split("\t", 1) for line in (base / "refs.tsv").read_text().splitlines() if line)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = {}
    report = open(out / "results.txt", "a")
    for spec in args.models:
        name, path = spec.split("=", 1)
        server = subprocess.Popen([args.server, "-m", path, "--host", "127.0.0.1", "--port", str(PORT), *args.server_args.split()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait()
            for tag, short in (("full context", False), ("short context", True)):
                hyps = run(items, short, args.lang)
                dump.setdefault(name, {})[tag] = {
                    "errors": [distance(normalize(r, args.lang), normalize(h, args.lang)) for (_, r), h in zip(items, hyps)],
                    "words": [len(normalize(r, args.lang)) for _, r in items],
                    "hyps": hyps,
                }
                line = f"{name}\t{tag}\t{len(items)} clips\t{score(items, hyps, args.lang)}"
                print(line, flush=True)
                report.write(line + "\n")
                report.flush()
        finally:
            server.terminate()
            server.wait()
        previous = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
        previous.update(dump)
        (out / "results.json").write_text(json.dumps(previous, ensure_ascii=False))


if __name__ == "__main__":
    main()
