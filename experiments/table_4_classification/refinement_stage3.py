"""Refinement evaluation, stage 3: incident-classification utility, on Haughton
et al.'s setup (the CECILIA incidents of Loya_NER_Cib.xlsx, 6 classes, their
70/30 split, seed 42; haughton_compare.incidents / split).

Text versions (no 4-type output anywhere):
  original
  manual_labels, manual_masking     the paper's pseudonymisation from hand annotation
  rubiner_<t>                       RubiNER, SIESTA's default keep rule (CVE, malware, ...)
  wp17_dslim_<t>, wp17_roberta_<t>  WP17's recipe trained on the 37 S-CECILIA seeds,
                                    native labels (preds_cecilia/ft_scec__*__native),
                                    replaced with SIESTA's own replace_entities; WP17
                                    has no keep rule, so every detected span is removed
  <t> = tokenisation, masking (+ pseudonymisation for RubiNER: per-type
  replacement needs its types)

Classifiers: the paper's two best classical pairs (XGBoost + bag of words,
XGBoost + TF-IDF; haughton_compare.classical settings) and RoBERTa-base, taken
as the paper's best transformer (batch 8, 5 seeds, model after the last
epoch; roberta_kit/).

Leak (x-axis of the utility plot): share of non-space characters of the
manual annotation's spans, on the 256 test incidents, that each version
leaves in clear. Group (hacker groups: public threat intelligence) is left
out, as in refinement_stage1.py.

Run from the repository root:
  SIESTA_DEVICE=0 .venv/bin/python experiments/table_4_classification/refinement_stage3.py anonymise
  .venv/bin/python experiments/table_4_classification/refinement_stage3.py classical
  .venv/bin/python experiments/table_4_classification/refinement_stage3.py kit      # then roberta_kit/run.py
  .venv/bin/python experiments/table_4_classification/refinement_stage3.py report
"""
import json, os, random, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
OUT = ROOT / "data_haughton" / "refinement"
KIT = ROOT / "experiments" / "table_4_classification" / "roberta_kit"
WP17 = {"wp17_dslim": "ft_scec__dslim__bert-base-NER__native",
        "wp17_roberta": "ft_scec__Jean-Baptiste__roberta-large-ner-english__native"}
VERSIONS = (["original", "manual_labels", "manual_masking",
             "rubiner_tokenisation", "rubiner_masking", "rubiner_pseudonymisation"]
            + [f"{w}_{t}" for w in WP17 for t in ("tokenisation", "masking")])
MODE = {"tokenisation": "label", "masking": "mask"}


def _docs():
    import haughton_compare as hc
    return hc.incidents()


def _wp17_spans(H):
    """WP17 predictions on CECILIA's text, carried to the Haughton text: 875
    texts are identical; 2 lack their first character in Loya_NER_Cib.xlsx
    (spans shifted by one); 3 rows hold an annotation export, not an
    incident, and get no spans (they are such rows in every version)."""
    os.environ["EVAL_DATASET"] = "cecilia"
    import corpora, score
    cec = {d["text"]: d["id"] for d in corpora.load()}
    out, notes = {}, {}
    for name, f in WP17.items():
        _, P = score.load_preds(ROOT / "preds_cecilia" / f"{f}.jsonl")
        spans = {}
        for h in H:
            t = h["text"]
            if t in cec:
                spans[h["id"]] = P.get(cec[t], [])
                continue
            full = next((c for c in cec if c[1:] == t), None)
            if full is not None:
                spans[h["id"]] = [(s - 1, e - 1, l) for s, e, l in P.get(cec[full], []) if s >= 1]
                notes[h["id"]] = "first character missing; spans shifted by one"
            else:
                spans[h["id"]] = []
                notes[h["id"]] = "annotation-export row, no incident text"
        out[name] = spans
    return out, notes


def _entities(text, spans):
    import scec_utility as su
    ents = su.gold_entities({"text": text, "spans": [list(x) for x in spans]})
    for e in ents:
        e["preserved"] = False   # WP17 keeps nothing
    return ents


def anonymise():
    import utility_rubiner as ur
    a = ur.build_rubiner()
    import pseudonymise   # date-overflow guard, as scec_utility.anonymise (fixC lacks the fix)
    overflow, _orig = [], pseudonymise.shift_date
    def _guarded(value, days):
        try:
            return _orig(value, days)
        except OverflowError:
            overflow.append(value); return None
    pseudonymise.shift_date = _guarded
    key = (ROOT / "logs" / "cecilia_utility_key.txt").read_text().strip()
    H = _docs()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"docs": len(H), "key": "logs/cecilia_utility_key.txt", "scope": "haughton", "versions": {}}
    # RubiNER: tokenisation and masking from data_haughton/rubiner (utility_rubiner run of 20 Oct),
    # pseudonymisation run here
    for t in ("tokenisation", "masking"):
        shutil.copy(ROOT / "data_haughton" / "rubiner" / f"{t}.jsonl", OUT / f"rubiner_{t}.jsonl")
        meta["versions"][f"rubiner_{t}"] = "data_haughton/rubiner (20 Oct)"
    n = 0
    with open(OUT / "rubiner_pseudonymisation.jsonl", "w", encoding="utf-8") as f:
        for d in H:
            _, ents, mod = a.anonymize(d["text"], 5, hash_key=key, scope="haughton")
            n += sum(not e.get("preserved") for e in ents)
            f.write(json.dumps({"id": d["id"], "text": mod}, ensure_ascii=False) + "\n")
    meta["versions"]["rubiner_pseudonymisation"] = {"redacted_entities": n, "date_overflow_values": list(overflow)}
    print("rubiner pseudonymisation", meta["versions"]["rubiner_pseudonymisation"], flush=True)
    spans, notes = _wp17_spans(H)
    meta["wp17_notes"] = notes
    for name in WP17:
        for t, mode in MODE.items():
            n = 0
            with open(OUT / f"{name}_{t}.jsonl", "w", encoding="utf-8") as f:
                for d in H:
                    ents = _entities(d["text"], spans[name][d["id"]])
                    n += len(ents)
                    mod = a.replace_entities(d["text"], ents, mode=mode, hash_key=key, scope="haughton")
                    f.write(json.dumps({"id": d["id"], "text": mod, "entities": [[e["start"], e["end"]] for e in ents]},
                                       ensure_ascii=False) + "\n")
            meta["versions"][f"{name}_{t}"] = {"redacted_entities": n, "predictions": f"preds_cecilia/{WP17[name]}.jsonl"}
            print(name, t, n, flush=True)
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)


def versions():
    H = _docs()
    v = {"original": [d["text"] for d in H], "manual_labels": [d["paper_labels"] for d in H],
         "manual_masking": [d["paper_masked"] for d in H]}
    for name in VERSIONS[3:]:
        by = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(OUT / f"{name}.jsonl", encoding="utf-8")}
        v[name] = [by[d["id"]] for d in H]
    return H, v


def _boot_diff(y, pa, pb, n=2000):
    """Paired bootstrap over test incidents of F1-w(b) - F1-w(a)."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0); m = len(y); d = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        d.append(f1_score(y[i], pb[i], average="weighted", zero_division=0)
                 - f1_score(y[i], pa[i], average="weighted", zero_division=0))
    d.sort()
    return [float(d[int(0.025 * n)]), float(d[int(0.975 * n)])]


def classical():
    import warnings
    import haughton_compare as hc
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, FunctionTransformer
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import f1_score, accuracy_score
    from xgboost import XGBClassifier
    warnings.filterwarnings("ignore")
    vec = {"BOW": lambda: CountVectorizer(min_df=5, ngram_range=(1, 2), max_features=5000),
           "TF-IDF": lambda: TfidfVectorizer(min_df=5, ngram_range=(1, 2), use_idf=True, smooth_idf=True, norm="l2")}
    H, v = versions()
    tr, te = hc.split(H, "6c")
    y_all = np.array([d["topic"] for d in H]); le = LabelEncoder().fit(y_all[np.r_[tr, te]])
    ytr, yte = le.transform(y_all[tr]), le.transform(y_all[te])
    res = {}
    for vn, vm in vec.items():
        model = f"XGBoost+{vn}"; res[model] = {}
        for name, texts in v.items():
            X = np.array(texts, dtype=object)
            p = Pipeline([("v", vm()), ("d", FunctionTransformer(lambda x: np.asarray(x.todense()), accept_sparse=True)),
                          ("c", XGBClassifier(eval_metric="mlogloss", random_state=hc.SEED))])
            pred = p.fit(X[tr], ytr, c__sample_weight=compute_sample_weight("balanced", ytr)).predict(X[te])
            res[model][name] = {"f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
                                "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
                                "accuracy": accuracy_score(yte, pred), "pred": pred.tolist()}
            print(model, f"{name:26s} {res[model][name]['f1_weighted']:.3f}", flush=True)
        base = np.array(res[model]["original"]["pred"])
        for name in v:
            if name != "original":
                res[model][name]["diff_vs_original_ci95"] = _boot_diff(yte, base, np.array(res[model][name]["pred"]))
    json.dump({"test_size": len(te), "y_test": yte.tolist(), "results": res},
              open(ROOT / "logs" / "refinement_stage3_classical.json", "w"))


def kit():
    """Write the text versions and the split into roberta_kit/data, for its run.py."""
    import haughton_compare as hc
    data = KIT / "data"; data.mkdir(exist_ok=True)
    H, v = versions()
    for name, texts in v.items():
        with open(data / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d, t in zip(H, texts):
                f.write(json.dumps({"id": d["id"], "label": d["topic"], "text": t}, ensure_ascii=False) + "\n")
    json.dump({s: dict(zip(("train", "test"), (x.tolist() for x in hc.split(H, s)))) for s in ("6c", "10c")},
              open(data / "split.json", "w"))
    print("kit:", KIT, "| versions:", len(v))


def leak():
    """Share of manual-annotation characters (6C test incidents) left in clear."""
    import haughton_compare as hc
    H = _docs(); _, te = hc.split(H, "6c")
    spans, _ = _wp17_spans(H)
    rub = {json.loads(l)["id"]: [(e["start"], e["end"]) for e in json.loads(l)["entities"] if not e.get("preserved")]
           for l in open(ROOT / "data_haughton" / "rubiner" / "tokenisation.jsonl", encoding="utf-8")}
    removed = {"rubiner": rub}
    for name in WP17:
        removed[name] = {d["id"]: [(e["start"], e["end"]) for e in _entities(d["text"], spans[name][d["id"]])] for d in H}
    out = {}
    for name, R in removed.items():
        g = c = 0
        for i in te:
            d = H[i]
            if d["gold"] is None:
                continue
            t = d["text"]; gm, rm = bytearray(len(t)), bytearray(len(t))
            for x in d["gold"]:
                if x["label"] == "Group":
                    continue
                s, e = hc._trim(t, x["start"], x["end"]); gm[s:e] = b"\1" * max(0, e - s)
            for s, e in R[d["id"]]:
                rm[s:e] = b"\1" * (e - s)
            for k, ch in enumerate(t):
                if not ch.isspace() and gm[k]:
                    g += 1; c += rm[k]
        out[name] = 1 - c / g
    return out


def report():
    lk = leak()
    cls = json.load(open(ROOT / "logs" / "refinement_stage3_classical.json"))["results"]
    tr = {}
    res_dir = KIT / os.environ.get("RESULTS", "results")
    for f in (res_dir / "roberta" / "6c").glob("*/seed*.json"):
        r = json.loads(f.read_text())
        tr.setdefault(r["text"], {})[r["seed"]] = r["metrics_last"]["f1_weighted"]
    rows = []
    sys_of = lambda n: "rubiner" if n.startswith("rubiner") else next((w for w in WP17 if n.startswith(w)), None)
    for name in VERSIONS:
        s = sys_of(name)
        row = {"version": name, "leak": (0.0 if name.startswith("manual") else 1.0 if name == "original" else lk[s]),
               **{m: cls[m][name]["f1_weighted"] for m in cls},
               **{f"{m} diff CI": cls[m][name].get("diff_vs_original_ci95") for m in cls}}
        seeds = tr.get(name, {})
        if seeds:
            base = tr.get("original", {})
            common = sorted(set(seeds) & set(base))
            diffs = [seeds[k] - base[k] for k in common]
            row.update({"RoBERTa mean": float(np.mean(list(seeds.values()))), "RoBERTa seeds": len(seeds),
                        "RoBERTa diff mean": float(np.mean(diffs)) if diffs else None,
                        "RoBERTa diff sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None})
        rows.append(row)
    json.dump(rows, open(ROOT / "logs" / "refinement_stage3_report.json", "w"), indent=1)
    for r in rows:
        print(f"{r['version']:26s} leak {r['leak']*100:5.1f}%  " + "  ".join(
            f"{m} {r[m]:.3f}" for m in cls) + (f"  RoBERTa {r['RoBERTa mean']:.3f} (n={r['RoBERTa seeds']})"
                                                if "RoBERTa mean" in r else ""))


if __name__ == "__main__":
    {"anonymise": anonymise, "classical": classical, "kit": kit, "report": report}[sys.argv[1]]()
