import random
import re
import unicodedata
from pathlib import Path

LANGS = {"sr_rs": "sr", "hr_hr": "hr", "en_us": "en"}
CYR = dict(zip("абвгдђежзијклмнопрстћуфхцчш", "a b v g d đ e ž z i j k l m n o p r s t ć u f h c č š".split()))
CYR.update({"љ": "lj", "њ": "nj", "џ": "dž"})


def latin(text):
    out = []
    for c in text:
        low = c.lower()
        if low in CYR:
            t = CYR[low]
            out.append(t if c == low else t[0].upper() + t[1:])
        else:
            out.append(c)
    return "".join(out)


def normalize(text, lang):
    text = unicodedata.normalize("NFC", text).lower()
    if lang == "sr":
        text = "".join(CYR.get(c, c) for c in text)
    return re.sub(r"[^\w\s']", " ", text.replace("’", "'")).split()


def distance(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def read_tsv(path):
    rows = [line.split("\t") for line in Path(path).read_text().splitlines()]
    return [r for r in rows if len(r) > 3]


def sample_test(tsv, count, seed=7):
    refs = {r[1]: r[3] for r in read_tsv(tsv)}
    names = sorted(refs)
    random.seed(seed)
    random.shuffle(names)
    return {n: refs[n] for n in names[:count]}
