import argparse
import io
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

REPO = "classla/ParlaSpeech-{}"
FILES = {"sr": ("RS", 138), "hr": ("HR", 326)}
URL = "https://huggingface.co/datasets/classla/ParlaSpeech-{}/resolve/main/data/{}"


def shard_name(index, total):
    return f"train-{index:05d}-of-{total:05d}.parquet"


def pick_shards(total, count):
    last = total - 4
    return sorted({round(i * last / max(1, count - 1)) for i in range(count)})


def test_rows(repo, total, refs):
    fs = HfFileSystem()
    path = f"datasets/{REPO.format(repo)}/data/{shard_name(total - 1, total)}"
    with fs.open(path) as f:
        table = pq.read_table(f, columns=["date", "speaker_name", "text_normalised"])
    return [r for r in table.to_pylist() if (r["text_normalised"] or "").strip() in refs], {r["date"] for r in table.to_pylist()}


def usable(row, dates, speakers, refs, seen):
    text = (row["text_normalised"] or "").strip()
    if row["date"] in dates or row["speaker_name"] in speakers:
        return "test date or speaker"
    if not text or "[[" in text or len(text) < 8 or text in refs or text in seen:
        return "text"
    return None


def collect(lang, shards, data, minimum, maximum):
    repo, total = FILES[lang]
    base = Path(data) / "train" / f"parla_{lang}"
    base.mkdir(parents=True, exist_ok=True)
    meta = base / "meta.tsv"
    refs_file = Path("data/speech") / lang / "parliament" / "refs.tsv"
    refs = {line.split("\t")[1].strip() for line in refs_file.read_text().splitlines()}
    rows, dates = test_rows(repo, total, refs)
    speakers = {r["speaker_name"] for r in rows}
    print(lang, "test shard dates", len(dates), "test speakers", len(speakers), flush=True)
    done = set()
    if meta.exists():
        done = {line.split("\t")[0] for line in meta.read_text().splitlines()}
    seen = set()
    for index in shards:
        tag = f"s{index:03d}"
        if tag in done:
            continue
        part = base / "part.parquet"
        urllib.request.urlretrieve(URL.format(repo, shard_name(index, total)), part)
        lines = []
        why = {}
        reader = pq.ParquetFile(part)
        columns = ["id", "audio", "text_normalised", "date", "speaker_name"]
        n = 0
        for batch in reader.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                n += 1
                reason = usable(row, dates, speakers, refs, seen)
                if reason is None:
                    audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
                    if rate != 16000 or audio.ndim != 1 or not minimum <= len(audio) / rate <= maximum:
                        reason = "length or format"
                if reason is not None:
                    why[reason] = why.get(reason, 0) + 1
                    continue
                text = " ".join(row["text_normalised"].split())
                seen.add(text)
                name = f"{tag}_{n:05d}"
                sf.write(base / f"{name}.flac", audio, 16000, subtype="PCM_16")
                lines.append(f"{tag}\t{name}.wav\t{text}\t{text}")
        part.unlink()
        with meta.open("a") as f:
            f.write("\n".join(lines) + "\n")
        size = sum(p.stat().st_size for p in base.glob("*.flac")) / 1e9
        print(f"{lang} shard {index}: kept {len(lines)} of {n} {why} folder {size:.2f} GB", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--sr", type=int, default=8, help="Serbian files, about 420 MB each")
    p.add_argument("--hr", type=int, default=2, help="Croatian files, about 550 MB each")
    p.add_argument("--min", type=float, default=2)
    p.add_argument("--max", type=float, default=28)
    args = p.parse_args()
    for lang, count in (("sr", args.sr), ("hr", args.hr)):
        if count:
            collect(lang, pick_shards(FILES[lang][1], count), args.data, args.min, args.max)


if __name__ == "__main__":
    main()
