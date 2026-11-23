import argparse
import math
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

from common import LANGS, latin, read_tsv

TEACHER = "openai/whisper-large-v3-turbo"
STUDENT = "openai/whisper-small"
TEXT_VOCAB = 50257
TOPK = 64
TEMP = 2.0
DEVICE = "mps"
PREFIX = 4


def battery():
    text = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    level = re.search(r"(\d+)%", text)
    return "AC Power" in text, int(level.group(1)) if level else 100


def low_power():
    ac, level = battery()
    return not ac and level < 20


def low_disk(work):
    return shutil.disk_usage(work).free < 1.5e9


def rows(root, folder, lang, limit, test_texts):
    base = root / folder
    wavs = {p.stem + ".wav": p for p in base.rglob("*.flac")}
    items = []
    skipped = 0
    for cols in read_tsv(base / "meta.tsv"):
        if cols[1] not in wavs:
            continue
        if cols[3].strip() in test_texts:
            skipped += 1
            continue
        text = latin(cols[2]) if lang == "sr" else cols[2]
        items.append({"wav": str(wavs[cols[1]]), "text": text.strip(), "lang": lang})
    random.Random(1).shuffle(items)
    print(folder, "usable", len(items), "skipped as test overlap", skipped, flush=True)
    return items[:limit]


def load_audio(path):
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    assert rate == 16000
    return audio[: 30 * 16000]


def tokens(tok, text, lang):
    tok.set_prefix_tokens(language=lang, task="transcribe", predict_timestamps=False)
    return tok(text).input_ids[:200]


def precompute(args, items):
    cache = Path(args.work) / "teacher.pt"
    if cache.exists():
        data = torch.load(cache)
        key = [(i["text"], i["lang"]) for i in items]
        if [(i["text"], i["lang"]) for i in data["items"]] == key:
            return {"items": items, "targets": data["targets"]}
        print("cached teacher targets do not match this data, recomputing", flush=True)
    tt = WhisperTokenizer.from_pretrained(TEACHER)
    fe = WhisperFeatureExtractor.from_pretrained(TEACHER)
    model = WhisperForConditionalGeneration.from_pretrained(TEACHER, dtype=torch.float16).to(DEVICE).eval()
    out = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(items), 8):
            if low_power() or low_disk(args.work):
                print("battery or disk low, stopping teacher pass", flush=True)
                break
            batch = items[i : i + 8]
            feats = fe([load_audio(b["wav"]) for b in batch], sampling_rate=16000, return_tensors="pt").input_features
            ids = [tokens(tt, b["text"], b["lang"]) for b in batch]
            width = (max(map(len, ids)) + 31) // 32 * 32
            dec = torch.full((len(ids), width), tt.eos_token_id)
            for j, t in enumerate(ids):
                dec[j, : len(t)] = torch.tensor(t)
            logits = model(input_features=feats.to(DEVICE, torch.float16), decoder_input_ids=dec.to(DEVICE)).logits.float()
            text_logits = logits[..., : TEXT_VOCAB + 1]
            top = text_logits.topk(TOPK, dim=-1)
            vals, idx = top.values.cpu(), top.indices.cpu()
            for j, t in enumerate(ids):
                n = len(t) - 1
                out.append({"vals": vals[j, 3:n].half(), "idx": idx[j, 3:n].int(), "n": n - 3})
            if i % 160 == 0:
                torch.mps.empty_cache()
                print(f"teacher {i + len(batch)}/{len(items)} {time.time() - start:.0f}s", flush=True)
    out = {"items": items[: len(out)], "targets": out}
    torch.save(out, cache)
    del model
    torch.mps.empty_cache()
    return out


def train(args, data):
    items, targets = data["items"], data["targets"]
    ts = WhisperTokenizer.from_pretrained(STUDENT)
    fe = WhisperFeatureExtractor.from_pretrained(STUDENT)
    model = WhisperForConditionalGeneration.from_pretrained(STUDENT, dtype=torch.float32).to(DEVICE)
    model.config.use_cache = False
    model.train()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    reserve = work / "reserve.bin"
    reserve.write_bytes(b"\1" * 700_000_000)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start = time.time()
    order = list(range(len(items)))
    rng = random.Random(3)
    step = 0
    losses = []
    done = False
    while not done:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            elapsed = time.time() - start
            if elapsed > args.minutes * 60 or (step % 20 == 0 and (low_power() or low_disk(args.work))):
                done = True
                break
            frac = elapsed / (args.minutes * 60)
            lr = args.lr * (min(1.0, step / 50)) * max(0.05, 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            full = [order[k] for k in range(i, i + args.batch)]
            size = args.batch // args.accum
            sums = []
            opt.zero_grad(set_to_none=True)
            try:
                for micro in [full[q : q + size] for q in range(0, args.batch, size)]:
                    pick = micro
                    feats = fe([load_audio(items[k]["wav"]) for k in pick], sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
                    ids = [tokens(ts, items[k]["text"], items[k]["lang"]) for k in pick]
                    width = (max(map(len, ids)) + 31) // 32 * 32
                    dec = torch.full((len(ids), width), ts.eos_token_id)
                    labels = torch.full((len(ids), width), -100)
                    kd_vals = torch.zeros(len(ids), width, TOPK)
                    kd_idx = torch.zeros(len(ids), width, TOPK, dtype=torch.long)
                    mask = torch.zeros(len(ids), width)
                    for j, t in enumerate(ids):
                        dec[j, : len(t)] = torch.tensor(t)
                        labels[j, PREFIX - 1 : len(t) - 1] = torch.tensor(t[PREFIX:])
                        tg = targets[pick[j]]
                        n = min(tg["n"], len(t) - PREFIX)
                        kd_vals[j, PREFIX - 1 : PREFIX - 1 + n] = tg["vals"][:n].float()
                        kd_idx[j, PREFIX - 1 : PREFIX - 1 + n] = tg["idx"][:n].long()
                        mask[j, PREFIX - 1 : PREFIX - 1 + n] = 1
                    logits = model(input_features=feats, decoder_input_ids=dec.to(DEVICE)).logits
                    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.to(DEVICE).reshape(-1), ignore_index=-100)
                    s = logits.gather(-1, kd_idx.to(DEVICE)) / TEMP
                    kl = F.kl_div(F.log_softmax(s, -1), F.softmax(kd_vals.to(DEVICE) / TEMP, -1), reduction="none").sum(-1)
                    m = mask.to(DEVICE)
                    kd = (kl * m).sum() / m.sum().clamp(min=1) * TEMP * TEMP
                    loss = args.alpha * kd + (1 - args.alpha) * ce
                    (loss / args.accum).backward()
                    sums.append((ce.item(), kd.item()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            except RuntimeError as err:
                print(f"stopped at step {step}: {str(err)[:200]}", flush=True)
                done = True
                break
            losses.append(tuple(np.mean(sums, axis=0)))
            step += 1
            if step % 10 == 0:
                torch.mps.empty_cache()
            if step % 25 == 0:
                c, k = np.mean(losses[-25:], axis=0)
                print(f"step {step} ce {c:.3f} kd {k:.3f} {step / (time.time() - start):.2f} it/s epoch {step * args.batch / len(items):.2f} mps {torch.mps.driver_allocated_memory() / 1e9:.1f}GB disk {shutil.disk_usage(args.work).free / 1e9:.1f}GB batt {battery()}", flush=True)
    model.eval().half()
    reserve.unlink()
    out = Path(args.work) / "student"
    model.save_pretrained(out, safe_serialization=True)
    ts.save_pretrained(out)
    fe.save_pretrained(out)
    print(f"done: {step} steps, {step * args.batch / len(items):.1f} epochs, saved {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train")
    p.add_argument("--work", default="work")
    p.add_argument("--test", default="data/test")
    p.add_argument("--minutes", type=float, default=35)
    p.add_argument("--sr", type=int, default=100000)
    p.add_argument("--hr", type=int, default=200)
    p.add_argument("--en", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    test_texts = {r[3].strip() for lang in LANGS for r in read_tsv(Path(args.test) / lang / "test.tsv")}
    root = Path(args.data)
    items = rows(root, "sr_rs", "sr", args.sr, test_texts) + rows(root, "hr_hr", "hr", args.hr, test_texts) + rows(root, "en_us", "en", args.en, test_texts)
    random.Random(2).shuffle(items)
    data = precompute(args, items)
    print("train clips", len(data["items"]), flush=True)
    train(args, data)


main()
