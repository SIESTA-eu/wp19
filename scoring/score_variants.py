"""Score prediction files with score.score on the deliverable's splits, without
rewriting any results_*.json. Writes scores/scores_<name>.json
({results file name: record}) and prints one line per corpus.

  .venv/bin/python scoring/score_variants.py <prediction name> [<name> ...]
e.g. aad-fixC, aad-fixC-ft-siesta-rep, ft-cecilia__Jean-Baptiste__roberta-large-ner-english
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
os.chdir(ROOT)
JOBS = [("results_scec_test.json", "scec", "test"), ("results_cecilia_test.json", "cecilia", "test"),
        ("results_cyner_all.json", "cyner", "all"), ("scec_val4", "scec", "val4")]
VAL_SEEDS = {"SEED_004", "SEED_023", "SEED_037", "SEED_041"}


def main(name):
    import importlib, corpus, score
    out = {}
    for key, cname, subset in JOBS:
        os.environ["EVAL_DATASET"] = cname
        importlib.reload(corpus)
        p = corpus.preds_dir() / f"{name}.jsonl"
        if not p.exists(): continue
        docs = {d["id"]: d for d in corpus.load()}
        sp = corpus.split() if cname in ("scec", "cecilia") else None
        if subset == "test": ids = sp["test"]
        elif subset == "val4": ids = [i for i in sp["train"] if sp["gold_seed"][i] in VAL_SEEDS]
        else: ids = list(docs)
        groups = sp["gold_seed"] if cname == "scec" and subset == "test" else None
        r = score.score(p, ids, docs, corpus.gold_taxonomy(), groups)
        out[key] = r
        lk = r.get("leak_private", r["leak"])
        print(f"{name:52} {key:34} P {r['detect']['p']:.3f} R {r['detect']['r']:.3f} exR {r['exact']['r']:.3f} "
              f"F1 {r['detect']['f1']:.3f} exF1 {r['exact']['f1']:.3f} leak {lk*100:.1f}%", flush=True)
    (ROOT / "scores").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "scores" / f"scores_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
