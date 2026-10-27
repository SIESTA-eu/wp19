"""Train/test splits, fixed in two files next to this one (document ids only).

split_s_cecilia.json  S-CECILIA-Gold, seed-grouped. The reports were generated
                      by an LLM from 57 seed incidents and reports from one
                      seed share a template, so a random report-level split
                      leaks: whole seeds are held out instead (~25% of reports).
                        test       reports whose seed is held out (247)
                        train      the other reports (727)
                        gold_seed  seed incident of every report
split_cecilia.json    CECILIA-10C-900, D17's own protocol: sklearn
                      train_test_split(test_size=0.2, random_state=42), then
                      1/8 of the rest for validation (train / val / test).

  python corpora/splits.py    rebuilds split_s_cecilia.json
"""
import collections, json, random
from pathlib import Path
import pandas as pd

import cecilia, s_cecilia

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S_CECILIA = HERE / "split_s_cecilia.json"
CECILIA = HERE / "split_cecilia.json"
TEST_SHARE = 0.25


def s_cecilia_split():
    return json.load(open(S_CECILIA))


def cecilia_split():
    if not CECILIA.exists():
        from sklearn.model_selection import train_test_split
        ids = [d["id"] for d in cecilia.load()[0]]
        rest, test = train_test_split(ids, test_size=0.2, random_state=42)
        train, val = train_test_split(rest, test_size=0.1 / 0.8, random_state=42)
        json.dump(dict(train=train, val=val, test=test), open(CECILIA, "w"), indent=1)
    return json.load(open(CECILIA))


def build_s_cecilia():
    recs = pd.read_csv(ROOT / "data" / "S-CECILIA.csv")
    seed_of = dict(zip(recs.Record_ID, recs.Seed_ID))
    g = s_cecilia.load()
    gcsv = pd.read_csv(ROOT / "data" / "S-CECILIA-NER-GOLD.csv").set_index("Gold_ID")
    gseed = {x["id"]: seed_of[gcsv.at[x["id"], "Record_ID"]] for x in g}

    per_seed = collections.Counter(gseed.values())
    seeds = sorted(per_seed); random.seed(42); random.shuffle(seeds)
    test_seeds, n = [], 0
    for s in seeds:
        if n >= TEST_SHARE * len(g): break
        test_seeds.append(s); n += per_seed[s]
    test = [x["id"] for x in g if gseed[x["id"]] in test_seeds]
    train = [x["id"] for x in g if gseed[x["id"]] not in test_seeds]
    return dict(test_seeds=sorted(test_seeds), test=test, train=train, gold_seed=gseed)


if __name__ == "__main__":
    out = build_s_cecilia()
    json.dump(out, open(S_CECILIA, "w"), indent=1)
    print(f"test seeds {len(out['test_seeds'])} | test {len(out['test'])} | train {len(out['train'])}")
