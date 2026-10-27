"""Choose the fix-7 cyber-model threshold from the S-CECILIA training seeds.

Rule, fixed before any held-out result was looked at (13-14 October): the
highest training detection F1 among siesta-fix567-t<x>, among those whose
private leak is at most 0.5 point above siesta-fix56's.

Reads logs/score_fix_scec_train.log (score.py scec train) and prints the
chosen threshold, e.g. 0.7, on the last line.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL = 0.005


def main(path=HERE / "logs" / "score_fix_scec_train.log"):
    lines = Path(path).read_text().splitlines()
    split = next(i for i, l in enumerate(lines) if l.startswith("system") and "priv leak" in l)
    f1, leak = {}, {}
    for l in lines[:split]:
        m = re.match(r"(siesta-fix\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", l)
        if m:
            f1[m.group(1)] = float(m.group(5))
    for l in lines[split:]:
        m = re.match(r"(siesta-fix\S+)\s.*\s([\d.]+)\s*$", l)
        if m:
            leak[m.group(1)] = float(m.group(2))
    base = leak["siesta-fix56"]
    cands = {v: f1[v] for v in f1 if v.startswith("siesta-fix567-t") and leak.get(v, 1) <= base + TOL}
    for v in sorted(f1):
        print(f"{v:22} F1 {f1[v]:.3f}  private leak {leak.get(v, float('nan')):.3f}"
              + ("  eligible" if v in cands else ""))
    best = max(cands, key=lambda v: (cands[v], -float(v.split("-t")[1])))
    print(best.split("-t")[1])


if __name__ == "__main__":
    main(*sys.argv[1:])
