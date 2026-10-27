"""SIESTA against the manual annotation of Haughton et al. (2026, NLPAICS,
"When Privacy Helps..."): the same CECILIA incidents, annotated by hand with
span labels and pseudonymised from those spans, against SIESTA's automatic
detection and replacement.

Data (data_haughton/source, from the paper's shared folder):
  Loya_NER_Cib.xlsx   880 incidents: TEXT, TOPIC, the annotation export (NER:
                      one annotation-tool row, with the spans) and the
                      paper's pseudonymised columns, built from those spans:
                      Description_with_labels / _xxxx / _fake_data
  The paper's original-text runs read a second file (Updated
  Ciberincidentes.xlsx) whose rows are in a different order, so its
  original-text and pseudonymised test sets are not the same incidents (168 of
  256 shared on 6C). Here every version comes from Loya_NER_Cib.xlsx, row by
  row, so all versions share one split.

The export's spans are offsets into the text as annotated; they are placed
on TEXT (see _gold), and SIESTA runs on TEXT, the text the paper's
pseudonymised columns were made from.

Commands:
  SIESTA_DEVICE=0 haughton_compare.py anonymise aad6c5b
  haughton_compare.py detection aad6c5b        # SIESTA spans against the manual spans
  haughton_compare.py classical aad6c5b        # the paper's ten classical pairs, its split
  haughton_compare.py export_kit aad6c5b       # transformers, kits/haughton_cls
"""
import json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data_haughton" / "source"
OUT = HERE / "data_haughton"
TECHNIQUES = {"tokenisation": 1, "masking": 2, "substitution": 3, "hashing": 4, "pseudonymisation": 5}
PAPER = {"paper_labels": "Description_with_labels", "paper_masked": "Description_with_xxxx",
         "paper_fake": "Description_with_fake_data"}
DROP_6C = ["Intento Intrusion", "Obtencion Informacion", "Contenido Abusivo", "Vulnerable"]  # dropped, as the paper does
SEED = 42


EXPORT = re.compile(r'^\d+[, ]"1"[, ]"[^"]*"[, ]\d+[, ]"(\[.*?\])"[, ][\d.]+[, ]"(.*?)(?:"[, ]"[\d\-T:.]+Z")?"?\s*$', re.S)


def _fold(t):
    """Same-length folding of the characters the export and TEXT spell
    differently (curly quotes), after repairing UTF-8 read as Latin-1."""
    try:
        t = t.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return t.translate(str.maketrans("\u2019\u2018\u201c\u201d\u2013", "''\"\"-"))


def _key(t):
    return re.sub(r"[^a-z0-9]", "", _fold(t).lower())[:80]


def _gold(text, export):
    """Manual spans placed on TEXT. Some export cells hold only the start of
    the text, so a span that does not sit at its offset in TEXT is looked up
    by its string, nearest occurrence first."""
    out, lost = [], 0
    for sp in export:
        s, e, st = sp["start"], sp["end"], sp["text"]
        if text[s:e] != st:
            needle = _fold(st.strip())
            hits = [h.start() for h in re.finditer(re.escape(needle), _fold(text))] if needle else []
            if not hits:
                lost += 1; continue
            s = min(hits, key=lambda h: abs(h - s)); e = s + len(needle)
        out.append({"start": s, "end": e, "label": sp["labels"][0]})
    return out, lost


def incidents():
    """The export row is CSV with quotes not always escaped, so it is matched
    by pattern. In 74 rows the export in the NER column belongs to another
    incident (the pseudonymised columns of those rows do match TEXT), so
    exports are paired with incidents by their text; an incident with no
    export of its own has gold None and is left out of detection scoring."""
    d = pd.read_excel(SRC / "Loya_NER_Cib.xlsx")
    exports = {}
    for raw in d["NER"]:
        m = EXPORT.match(raw)
        exports.setdefault(_key(m.group(2).replace('""', '"')), json.loads(m.group(1).replace('""', '"')))
    docs = []
    for i, r in d.iterrows():
        text = str(r["TEXT"])
        exp = exports.get(_key(text))
        gold, lost = _gold(text, exp) if exp is not None else (None, 0)
        docs.append({"id": f"H-{i:03d}", "topic": r["TOPIC"], "text": text, "gold": gold, "gold_lost": lost,
                     **{k: str(r[c]) for k, c in PAPER.items()}})
    return docs


def anonymise(variant):
    sys.path.insert(0, str(HERE))
    import cecilia_utility as cu
    a = cu.build_archived(variant)
    import pseudonym
    key = (HERE / "logs" / "cecilia_utility_key.txt").read_text().strip()
    docs = incidents()
    out_dir = OUT / variant; out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for tech, opt in TECHNIQUES.items():
        t0, n_ent = time.time(), 0
        with open(out_dir / f"{tech}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                _, ents, mod = a.anonymize(d["text"], opt, hash_key=key, scope="haughton")
                n_ent += sum(not e.get("preserved") for e in ents)
                rec = {"id": d["id"], "text": mod}
                if tech == "tokenisation":
                    rec["entities"] = [{k: e.get(k) for k in ("start", "end", "text", "readable_type", "preserved")}
                                       for e in ents]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[tech] = {"seconds": round(time.time() - t0, 1), "redacted_entities": n_ent}
        print(variant, tech, stats[tech], flush=True)
    json.dump({"variant": variant, "docs": len(docs), "input": "TEXT", "techniques": stats},
              open(out_dir / "meta.json", "w"), indent=1)


def _trim(text, s, e):
    while s < e and not text[s].isalnum() and text[s] not in "$€£#@+(": s += 1
    while e > s and not text[e - 1].isalnum() and text[e - 1] not in ")%": e -= 1
    return s, e


def detection(variant):
    """Recall of each manual label: a gold span counts as found when SIESTA
    redacted text overlapping it ("any"), or a span with IoU >= 0.5 ("half").
    Preserved SIESTA detections (CVE, malware, ...) are not redactions and do
    not count. Precision: share of SIESTA redactions that overlap any gold
    span (the manual scheme has no types for e-mails, IPs, user names or
    domains, so they appear as SIESTA-only and are listed separately)."""
    docs = {d["id"]: d for d in incidents()}
    pred = {json.loads(l)["id"]: json.loads(l)["entities"] for l in open(OUT / variant / "tokenisation.jsonl")}
    found = defaultdict(lambda: [0, 0, 0]); siesta_types = defaultdict(Counter)
    unmatched_pred = Counter(); matched_pred = Counter(); examples = defaultdict(list)
    for i, d in docs.items():
        if d["gold"] is None:
            continue
        t = d["text"]
        gold = [(*_trim(t, g["start"], g["end"]), g["label"]) for g in d["gold"]]
        gold = [g for g in gold if g[1] > g[0]]
        reds = [(e["start"], e["end"], e["readable_type"]) for e in pred[i] if not e.get("preserved")]
        for s, e, lab in gold:
            ov = [(ps, pe, pt) for ps, pe, pt in reds if ps < e and pe > s]
            best = max((min(e, pe) - max(s, ps)) / (max(e, pe) - min(s, ps)) for ps, pe, _ in ov) if ov else 0
            found[lab][0] += 1; found[lab][1] += bool(ov); found[lab][2] += best >= 0.5
            if ov: siesta_types[lab][ov[0][2]] += 1
            elif len(examples[lab]) < 12: examples[lab].append(t[s:e])
        for ps, pe, pt in reds:
            (matched_pred if any(ps < e and pe > s for s, e, _ in gold) else unmatched_pred)[pt] += 1
    rows = []
    for lab, (n, a, h) in sorted(found.items(), key=lambda x: -x[1][0]):
        rows.append({"label": lab, "gold": n, "recall_any": a / n, "recall_half": h / n,
                     "siesta_types": dict(siesta_types[lab].most_common(4)), "missed_examples": examples[lab]})
        print(f"{lab:14} n={n:5}  any {a/n:.3f}  IoU>=.5 {h/n:.3f}  as {dict(siesta_types[lab].most_common(3))}")
    tm, tu = sum(matched_pred.values()), sum(unmatched_pred.values())
    print(f"SIESTA redactions overlapping a manual span: {tm}/{tm+tu} = {tm/(tm+tu):.3f}")
    print("SIESTA-only redactions by type:", dict(unmatched_pred.most_common(15)))
    json.dump({"per_label": rows, "redactions_matched": tm, "redactions_unmatched": tu,
               "unmatched_by_type": dict(unmatched_pred), "matched_by_type": dict(matched_pred)},
              open(HERE / "logs" / f"haughton_detection_{variant}.json", "w"), indent=1, ensure_ascii=False)


def versions(variant):
    docs = incidents()
    v = {"original": [d["text"] for d in docs]}
    v.update({k: [d[k] for d in docs] for k in PAPER})
    for tech in TECHNIQUES:
        by_id = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(OUT / variant / f"{tech}.jsonl")}
        v[f"siesta_{tech}"] = [by_id[d["id"]] for d in docs]
    return docs, v


def split(docs, scheme):
    """The paper's split: drop the four rarest classes for 6C, then 70/30
    stratified, seed 42, on the rows of Loya_NER_Cib.xlsx."""
    from sklearn.model_selection import train_test_split
    keep = np.array([scheme == "10c" or d["topic"] not in DROP_6C for d in docs])
    idx = np.flatnonzero(keep); y = np.array([d["topic"] for d in docs])[idx]
    tr, te = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=SEED, stratify=y)
    return idx[tr], idx[te]


def classical(variant):
    """The paper's ten classifier-vectoriser pairs with its settings
    (the final ML_Classifiers_Rewritten.ipynb in the paper's results folder: bag of
    words with min_df=5; an earlier draft in 02. EvaluationCode has min_df=100),
    fitted fresh for every version."""
    import warnings
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, FunctionTransformer
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from xgboost import XGBClassifier
    warnings.filterwarnings("ignore")
    vec = {"BOW": lambda: CountVectorizer(min_df=5, ngram_range=(1, 2), max_features=5000),
           "TF-IDF": lambda: TfidfVectorizer(min_df=5, ngram_range=(1, 2), use_idf=True, smooth_idf=True, norm="l2")}
    clf = {"SVM": lambda: SVC(kernel="linear", class_weight="balanced"),
           "LR": lambda: LogisticRegression(class_weight="balanced", penalty="l2", C=1.0, solver="sag", max_iter=2000),
           "RandomForest": lambda: RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=SEED),
           "XGBoost": lambda: XGBClassifier(eval_metric="mlogloss", random_state=SEED),
           "NaiveBayes": lambda: MultinomialNB()}
    docs, v = versions(variant)
    res = []
    for scheme in ("6c", "10c"):
        tr, te = split(docs, scheme)
        y_all = np.array([d["topic"] for d in docs]); le = LabelEncoder().fit(y_all[np.r_[tr, te]])
        ytr, yte = le.transform(y_all[tr]), le.transform(y_all[te])
        for name, texts in v.items():
            X = np.array(texts, dtype=object)
            for cn, cm in clf.items():
                for vn, vm in vec.items():
                    p = Pipeline([("v", vm()), ("d", FunctionTransformer(lambda x: np.asarray(x.todense()), accept_sparse=True)),
                                  ("c", cm())])
                    kw = {"c__sample_weight": compute_sample_weight("balanced", ytr)} if cn in ("XGBoost", "NaiveBayes") else {}
                    pred = p.fit(X[tr], ytr, **kw).predict(X[te])
                    _, _, f1w, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
                    _, _, f1m, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
                    res.append({"scheme": scheme, "text": name, "model": f"{cn}+{vn}", "f1_weighted": f1w,
                                "f1_macro": f1m, "accuracy": accuracy_score(yte, pred), "pred": pred.tolist()})
            print(scheme, f"{name:26}", " ".join(f"{r['f1_weighted']:.3f}" for r in res[-10:]), flush=True)
    json.dump(res, open(HERE / "logs" / f"haughton_classical_{variant}.json", "w"))


def export_kit(variant):
    kit = HERE / "kits" / "haughton_cls" / "data"; kit.mkdir(parents=True, exist_ok=True)
    docs, v = versions(variant)
    for name, texts in v.items():
        with open(kit / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d, t in zip(docs, texts):
                f.write(json.dumps({"id": d["id"], "label": d["topic"], "text": t}, ensure_ascii=False) + "\n")
    json.dump({s: dict(zip(("train", "test"), (x.tolist() for x in split(docs, s)))) for s in ("6c", "10c")},
              open(kit / "split.json", "w"))
    import shutil; shutil.copy(OUT / variant / "meta.json", kit / "anonymisation_meta.json")
    print("kit data:", sorted(p.name for p in kit.iterdir()))


if __name__ == "__main__":
    {"anonymise": anonymise, "detection": detection, "classical": classical,
     "export_kit": export_kit}[sys.argv[1]](sys.argv[2])
