import argparse
import json
import random


def paired(base, new, rounds=2000, seed=0):
    errors_a, errors_b, words = base["errors"], new["errors"], base["words"]
    assert new["words"] == words, "the two runs did not score the same clips"
    rng = random.Random(seed)
    n = len(words)
    diffs = []
    for _ in range(rounds):
        pick = [rng.randrange(n) for _ in range(n)]
        total = sum(words[i] for i in pick)
        diffs.append(100 * (sum(errors_b[i] for i in pick) - sum(errors_a[i] for i in pick)) / total)
    diffs.sort()
    wer = lambda e: 100 * sum(e) / sum(words)
    better = sum(b < a for a, b in zip(errors_a, errors_b))
    worse = sum(b > a for a, b in zip(errors_a, errors_b))
    return wer(errors_a), wer(errors_b), diffs[int(rounds * 0.025)], diffs[int(rounds * 0.975)], better, worse


def main():
    p = argparse.ArgumentParser(description="compare two models clip by clip from speech_tests.py --dump files")
    p.add_argument("base", help="file:model, e.g. results/v2_sr.json:v2")
    p.add_argument("new", help="file:model")
    args = p.parse_args()
    (fa, ma), (fb, mb) = (s.rsplit(":", 1) for s in (args.base, args.new))
    a, b = json.load(open(fa))[ma], json.load(open(fb))[mb]
    print(f"test\tclips\t{ma}\t{mb}\tchange (95%)\tbetter/worse clips")
    for tag in a:
        if tag not in b:
            continue
        wa, wb, lo, hi, better, worse = paired(a[tag], b[tag])
        print(f"{tag}\t{len(a[tag]['words'])}\t{wa:.1f}\t{wb:.1f}\t{wb - wa:+.1f} ({lo:+.1f} to {hi:+.1f})\t{better}/{worse}")


if __name__ == "__main__":
    main()
