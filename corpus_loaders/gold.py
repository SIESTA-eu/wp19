"""Load S-CECILIA gold: align CoNLL tokens to the raw text for char spans."""
import json, re, pandas as pd
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
DATA = ROOT / "data"


def read_conll(path):
    docs, cur, meta = [], None, None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("-DOCSTART-"):
            if cur is not None:
                docs.append((meta, cur))
            meta = line.split("#", 1)[1].strip().split()[0]
            cur = []
        elif line.strip():
            tok, tag = line.rsplit("\t", 1)
            cur.append((tok, tag))
    docs.append((meta, cur))
    return docs


def align(text, tokens):
    """Char offsets for each token, scanning left to right."""
    offs, pos = [], 0
    for tok in tokens:
        i = text.find(tok, pos)
        if i < 0 or text[pos:i].strip():
            return None
        offs.append((i, i + len(tok)))
        pos = i + len(tok)
    return offs


def spans_from_bio(tags, offs):
    spans, cur = [], None
    for tag, (s, e) in zip(tags, offs):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[2] != tag[2:])):
            if cur: spans.append(tuple(cur))
            cur = [s, e, tag[2:]]
        elif tag.startswith("I-"):
            cur[1] = e
        else:
            if cur: spans.append(tuple(cur)); cur = None
    if cur: spans.append(tuple(cur))
    return spans


def load():
    g = pd.read_csv(DATA / "S-CECILIA-NER-GOLD.csv")
    g = g[~g.Is_Duplicate_Annotation].set_index("Gold_ID")
    out = []
    for gid, toks in read_conll(DATA / "S-CECILIA-NER-GOLD.conll"):
        if gid not in g.index:
            continue
        text = g.at[gid, "text"]
        words = [t for t, _ in toks]; tags = [t for _, t in toks]
        offs = align(text, words)
        if offs is None:
            raise ValueError(f"cannot align {gid}")
        out.append(dict(id=gid, text=text, words=words, tags=tags, offs=offs,
                        spans=spans_from_bio(tags, offs), cls=g.at[gid, "Class"]))
    return out
