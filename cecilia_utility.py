"""Utility of anonymised text for incident classification: CECILIA-10C-900
anonymised by SIESTA with each technique, then classified as in the CECILIA
paper (Delgado et al., 2024, NLPAICS, "CECILIA: Enhancing CSIRT
Effectiveness with Transformer-Based Cyber Incident Classification").

Data: ../cecilia100_fixed_final.csv (880 incidents; 879 with a TOPIC). Labels:
  10C  the ten INCIBE classes (TOPIC)
  6C   the paper's CECILIA-6C-900: the four least represented classes
       (Vulnerable, Contenido Abusivo, Obtencion Informacion, Intento
       Intrusion) merged into Otros

Text versions (anonymise): the original, and SIESTA's five techniques on the
same detections -- tokenisation ([PER]), masking (xxxx), substitution (fake
values), hashing (keyed hash_...), pseudonymisation (per-type, keyed, date
shift) -- with the default preserved types (CVE, MALWARE, HASH, TECH,
SOFTWARE, OPERATING_SYSTEM kept). SIESTA is built through
run_siesta.build(<variant>), so the configuration is named and reproducible;
one fixed experiment key (logs/cecilia_utility_key.txt) for hashing and
pseudonymisation, scope "cecilia-utility".

Protocol (baselines): stratified 5-fold cross-validation, shuffled, seed 42,
the same folds for every text version; the paper's two traditional
classifiers, Logistic Regression on TF-IDF and k-NN on bag of words
(scikit-learn defaults, as the paper gives no settings); the paper's metrics:
accuracy, precision / recall / F1 weighted and macro, MCC, mean over folds.
The six transformers run on a server (kits/cecilia_cls).

Usage:
  SIESTA_DEVICE=0 cecilia_utility.py anonymise aad6c5b    # GitLab main (v2.1 detection, current replacement)
  SIESTA_DEVICE=0 cecilia_utility.py anonymise v2.1       # a2893cf
  SIESTA_DEVICE=0 cecilia_utility.py anonymise siesta-fix567-t0.7
  cecilia_utility.py baselines siesta-fix567-t0.7
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_cecilia_utility"
TECHNIQUES = {"tokenisation": 1, "masking": 2, "substitution": 3, "hashing": 4, "pseudonymisation": 5}
MERGE_6C = {"Vulnerable", "Contenido Abusivo", "Obtencion Informacion", "Intento Intrusion"}
SEED, FOLDS = 42, 5


def incidents():
    d = pd.read_csv(HERE.parent / "cecilia100_fixed_final.csv")
    d = d[d["TOPIC"].notna()].reset_index(drop=True)
    return [{"id": f"CEC-{int(r['Unnamed: 0']):04d}", "text": str(r["TEXT"]), "label_10c": r["TOPIC"],
             "label_6c": "Otros" if r["TOPIC"] in MERGE_6C else r["TOPIC"], "subtopic": r.get("SUBTOPIC")}
            for _, r in d.iterrows()]


# Clean copies made with git archive (repo and working tree untouched):
ARCHIVES = {"v2.1": HERE / "siesta_v2.1_a2893cf" / "Anonimizacion",   # a2893cf, the deliverable's detection run
            "aad6c5b": HERE / "siesta_aad6c5b" / "Anonimizacion"}     # GitLab main on 14 Oct: v2.1 detection, current replacement


def build_archived(variant):
    """SIESTA exactly as committed, from a clean copy, on the GPU as
    run_siesta.build does for release variants."""
    src = ARCHIVES[variant]
    sys.path.insert(0, str(src)); os.chdir(src)
    os.environ.pop("GLINER_ENABLED", None)
    import torch, custom_entities
    from en_anonymizer import EnglishAnonymizer
    if int(os.environ.get("SIESTA_DEVICE", "-1")) >= 0 and custom_entities._load() is not None:
        custom_entities._model = custom_entities._model.to("cuda")
    a = EnglishAnonymizer()
    if int(os.environ.get("SIESTA_DEVICE", "-1")) >= 0:
        for pipe in list(a.pipelines.values()) + ([a.cisco_pipe] if a.cisco_pipe else []):
            pipe.model.to("cuda"); pipe.device = torch.device("cuda")
    return a


def anonymise(variant):
    if variant in ARCHIVES:
        a = build_archived(variant)
        import pseudonym
    else:
        sys.path.insert(0, str(HERE))
        import run_siesta as rs
        os.environ.pop("GLINER_ENABLED", None)
        import custom_entities, pseudonym
        if rs.DEV >= 0 and custom_entities._load() is not None:
            custom_entities._model = custom_entities._model.to("cuda")
        a = rs.build(variant)
    keyf = HERE / "logs" / "cecilia_utility_key.txt"
    if not keyf.exists():
        keyf.write_text(pseudonym.new_key())
    key = keyf.read_text().strip()
    docs = incidents()
    out_dir = OUT / variant; out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "original.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats = {}
    for tech, opt in TECHNIQUES.items():
        t0, n_ent = time.time(), 0
        with open(out_dir / f"{tech}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                _, ents, mod = a.anonymize(d["text"], opt, hash_key=key, scope="cecilia-utility")
                n_ent += sum(not e.get("preserved") for e in ents)
                f.write(json.dumps({**d, "text": mod}, ensure_ascii=False) + "\n")
        stats[tech] = {"seconds": round(time.time() - t0, 1), "redacted_entities": n_ent}
        print(variant, tech, stats[tech], flush=True)
    json.dump({"variant": variant, "docs": len(docs), "techniques": stats}, open(out_dir / "meta.json", "w"), indent=1)


def folds(labels):
    from sklearn.model_selection import StratifiedKFold
    # the two classes with 2 incidents cannot be split 5 ways; StratifiedKFold warns and spreads them
    return list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))


def metrics(y, p):
    from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
    out = {"accuracy": accuracy_score(y, p), "mcc": matthews_corrcoef(y, p)}
    for avg in ("weighted", "macro"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=avg, zero_division=0)
        out.update({f"prec_{avg}": pr, "recall_{avg}".format(avg=avg): rc, f"f1_{avg}": f1})
    return out


def baselines(variant):
    import warnings
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    warnings.filterwarnings("ignore", message="The least populated class")
    models = {"LR TF-IDF": lambda: make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
              "KNN BoW": lambda: make_pipeline(CountVectorizer(), KNeighborsClassifier())}
    res = []
    for scheme in ("10c", "6c"):
        for tech in ["original"] + list(TECHNIQUES):
            docs = [json.loads(l) for l in open(OUT / variant / f"{tech}.jsonl", encoding="utf-8")]
            X, y = np.array([d["text"] for d in docs], dtype=object), np.array([d[f"label_{scheme}"] for d in docs])
            for name, make in models.items():
                per_fold = []
                for tr, te in folds(y):
                    m = make().fit(X[tr], y[tr]); per_fold.append(metrics(y[te], m.predict(X[te])))
                avg = {k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
                avg["accuracy_var"] = float(np.var([f["accuracy"] for f in per_fold]))
                res.append({"scheme": scheme, "text": tech, "model": name, **avg})
                print(f"{scheme} {tech:16} {name:9} acc {avg['accuracy']:.4f} F1-w {avg['f1_weighted']:.4f} "
                      f"F1-m {avg['f1_macro']:.4f} MCC {avg['mcc']:.4f}", flush=True)
    json.dump(res, open(HERE / "logs" / f"cecilia_utility_baselines_{variant}.json", "w"), indent=1)


def export_kit(variant):
    """Write kits/cecilia_cls/data: the six text versions and the folds
    (indices into the document order, from folds(), the same splits
    baselines() uses)."""
    import shutil
    kit = HERE / "kits" / "cecilia_cls" / "data"; kit.mkdir(parents=True, exist_ok=True)
    for tech in ["original"] + list(TECHNIQUES):
        shutil.copy(OUT / variant / f"{tech}.jsonl", kit / f"{tech}.jsonl")
    shutil.copy(OUT / variant / "meta.json", kit / "anonymisation_meta.json")
    docs = [json.loads(l) for l in open(OUT / variant / "original.jsonl", encoding="utf-8")]
    out = {}
    for scheme in ("10c", "6c"):
        y = np.array([d[f"label_{scheme}"] for d in docs])
        out[scheme] = [{"train": tr.tolist(), "test": te.tolist()} for tr, te in folds(y)]
    json.dump(out, open(kit / "folds.json", "w"))
    print("kit data written:", sorted(p.name for p in kit.iterdir()))


if __name__ == "__main__":
    {"anonymise": anonymise, "baselines": baselines, "export_kit": export_kit}[sys.argv[1]](sys.argv[2])
