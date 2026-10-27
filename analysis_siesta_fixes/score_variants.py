"""Score prediction files with score.score on the deliverable's splits, without
rewriting any results_*.json. Writes analysis_siesta_fixes/scores_<name>.json
({results file name: record}) and prints one line per corpus.

  .venv/bin/python analysis_siesta_fixes/score_variants.py <prediction name> [<name> ...]
e.g. aad-fixC, aad-fixC-appno (TAB only), ft-cecilia__Jean-Baptiste__roberta-large-ner-english

Also adds TAB's official ER_di / ER_qi / WP when a TAB prediction exists.
"""
import io, json, os, sys
from contextlib import redirect_stdout
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL)); os.chdir(EVAL)
JOBS = [("results_scec_test.json", "scec", "test"), ("results_cecilia_test.json", "cecilia", "test"),
        ("results_tab_all.json", "tab", "all"), ("results_cyner_all.json", "cyner", "all"),
        ("results_cyberpii_all.json", "cyberpii", "all"), ("results_casie_all.json", "casie", "all"),
        ("results_anonlfi_cases_case1.json", "anonlfi_cases", "case1"),
        ("results_anonlfi_cases_case2.json", "anonlfi_cases", "case2"),
        ("scec_val4", "scec", "val4")]
VAL_SEEDS = {"SEED_004", "SEED_023", "SEED_037", "SEED_041"}


def main(name):
    import importlib, corpora, score
    out = {}
    for key, corpus, subset in JOBS:
        os.environ["EVAL_DATASET"] = corpus
        importlib.reload(corpora)
        p = corpora.preds_dir() / f"{name}.jsonl"
        if not p.exists(): continue
        docs = {d["id"]: d for d in corpora.load()}
        sp = corpora.split() if corpus in ("scec", "cecilia") else None
        if corpus == "anonlfi_cases": ids = [i for i in docs if i.startswith(subset)]
        elif subset == "test": ids = sp["test"]
        elif subset == "val4": ids = [i for i in sp["train"] if sp["gold_seed"][i] in VAL_SEEDS]
        else: ids = list(docs)
        groups = sp["gold_seed"] if corpus == "scec" and subset == "test" else None
        r = score.score(p, ids, docs, corpora.gold_taxonomy(), groups)
        out[key] = r
        lk = r.get("leak_private", r["leak"])
        print(f"{name:52} {key:34} P {r['detect']['p']:.3f} R {r['detect']['r']:.3f} exR {r['exact']['r']:.3f} "
              f"F1 {r['detect']['f1']:.3f} exF1 {r['exact']['f1']:.3f} leak {lk*100:.1f}%", flush=True)
    tabp = EVAL / "preds_tab" / f"{name}.jsonl"
    if tabp.exists():
        sys.path.insert(0, str(EVAL / "third_party" / "TAB"))
        import evaluation as tab
        with redirect_stdout(io.StringIO()):
            gold = tab.GoldCorpus(str(EVAL / "third_party" / "TAB" / "echr_test.json"))
            lines = [json.loads(l) for l in open(tabp)][1:]
            m = [tab.MaskedDocument(o["id"], [(s, e) for s, e, _ in o["spans"]]) for o in lines]
            off = dict(ER_di=gold.get_entity_recall(m, True, False), ER_qi=gold.get_entity_recall(m, False, True),
                       WP_diqi=gold.get_precision(m, tab.UniformTokenWeighting(), token_level=True))
        out["tab_official"] = off
        print(f"{name:52} TAB official: ER_di {off['ER_di']:.3f} ER_qi {off['ER_qi']:.3f} WP {off['WP_diqi']:.3f}")
    json.dump(out, open(EVAL / "analysis_siesta_fixes" / f"scores_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
