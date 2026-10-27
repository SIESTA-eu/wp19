"""Load CECILIA as WP17 used it: cecilia100_fixed_final.csv.

This is the file WP17/Task 17.2/ner-finetune.py reads. Its INFO column pairs
every incident with its own Label Studio annotations. (CECILIA-10C-900-NER.xlsx
does not: from about row 630 its annotation column is shifted by one row. The
"fixed" file is that join repaired.) It agrees span-for-span with the
original Label Studio exports on 839 of the 854 incidents both contain, and
has 25 annotated incidents the exports lack.

Rows are kept exactly as WP17's prepare_data() keeps them (INFO parses to a
non-empty list), in file order, so splits.cecilia_split() -- WP17's own
train_test_split calls -- reproduces D17.1's test set.

In 74 rows, 275 spans were annotated on mojibake text ("VallÃ¨s") while TEXT
is clean ("Vallès"), so their offsets drift. They are re-anchored: the
repaired string nearest the recorded offset. Label "0" (WP17's punctuation
marker) is skipped, as in WP17.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fix(t):
    try:
        return t.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return t


def _anchor(text, a):
    if text[a["start"]:a["end"]] == a["text"]:
        return a["start"], a["end"]
    for cand in (fix(a["text"]), a["text"]):
        c = cand.strip()
        if not c: continue
        hits, j = [], text.find(c)
        while j >= 0:
            hits.append(j); j = text.find(c, j + 1)
        if hits:
            s = min(hits, key=lambda h: abs(h - a["start"]))
            return s, s + len(c)
    return None


def load():
    df = pd.read_csv(ROOT / "data_cecilia" / "cecilia100_fixed_final.csv")
    docs, stats = [], dict(rows=len(df), kept=0, reanchored=0, lost=0)
    for i, r in df.iterrows():
        try:
            ann = json.loads(str(r["INFO"]))
        except (ValueError, TypeError):
            continue
        ann = [a for a in ann if isinstance(a, dict) and all(k in a for k in ("start", "end", "text", "labels"))]
        if not ann:
            continue
        text = str(r["TEXT"]); spans = []
        for a in ann:
            label = (a["labels"] or ["0"])[0]
            if label == "0": continue
            pos = _anchor(text, a)
            if pos is None: stats["lost"] += 1; continue
            if pos != (a["start"], a["end"]) and text[a["start"]:a["end"]] != a["text"]: stats["reanchored"] += 1
            s, e = pos
            while s < e and text[s].isspace(): s += 1
            while e > s and text[e - 1].isspace(): e -= 1
            if e > s: spans.append((s, e, label))
        docs.append(dict(id=f"CEC-{i:04d}", text=text, spans=sorted(set(spans)), cls=r["TOPIC"]))
        stats["kept"] += 1
    return docs, stats


if __name__ == "__main__":
    import collections
    d, st = load()
    c = collections.Counter(s[2] for x in d for s in x["spans"])
    print(len(d), "docs,", sum(c.values()), "spans |", st)
    for k, v in c.most_common(): print(f"  {k:16}{v}")
