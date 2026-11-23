import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

from common import LANGS, sample_test

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
TRAIN = {"sr_rs": "train", "hr_hr": "dev", "en_us": "dev"}


def stream(lang, split, wanted, save):
    with urllib.request.urlopen(f"{BASE}/{lang}/audio/{split}.tar.gz") as res, tarfile.open(fileobj=res, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name.endswith(".wav") and (wanted is None or name in wanted):
                audio, rate = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                assert rate == 16000
                save(name, audio)


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(urllib.request.urlopen(url).read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--clips", type=int, default=100)
    p.add_argument("--langs", default=",".join(LANGS))
    p.add_argument("--only", choices=["train", "test"])
    args = p.parse_args()
    for lang in args.langs.split(","):
        if args.only != "test":
            base = Path(args.data) / "train" / lang
            fetch(f"{BASE}/{lang}/{TRAIN[lang]}.tsv", base / "meta.tsv")
            stream(lang, TRAIN[lang], None, lambda n, a: sf.write(base / (Path(n).stem + ".flac"), a, 16000, subtype="PCM_16"))
            print(lang, "train clips", len(list(base.rglob("*.flac"))), flush=True)
        if args.only != "train":
            base = Path(args.data) / "test" / lang
            fetch(f"{BASE}/{lang}/test.tsv", base / "test.tsv")
            wanted = set(sample_test(base / "test.tsv", args.clips))
            (base / "wav").mkdir(exist_ok=True)
            stream(lang, "test", wanted, lambda n, a: sf.write(base / "wav" / n, a, 16000, subtype="PCM_16"))
            print(lang, "test clips", len(list((base / "wav").glob("*.wav"))), flush=True)


main()
