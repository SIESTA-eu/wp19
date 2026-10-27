"""Utility of anonymised S-CECILIA reports for incident classification:
SIESTA's automatic detection against perfect detection (the gold spans), each
replaced by SIESTA's own replacement code, so the two differ in detection only.

Data: S-CECILIA gold (s_cecilia.load(), 974 reports, 10 classes in "cls"; each
report derives from one seed incident, corpora/split_s_cecilia.json "gold_seed"; every seed has
one class and every class has at least two seeds).

Text versions:
  original
  gold_<technique>    gold spans, typed onto SIESTA's catalogue (GOLD_TO_SIESTA)
                      and replaced by SIESTA aad6c5b's replace_entities /
                      replace_with_fake_data, with SIESTA's default policy:
                      CVE, MALWARE, HASH, SOFTWARE, OPERATING_SYSTEM kept
  siesta_<technique>  SIESTA aad6c5b, automatic detection, default settings
for the five techniques: tokenisation, masking, substitution, hashing,
pseudonymisation (same key as the other utility studies, scope "scec-utility").

Folds: grouped 5-fold cross-validation by seed incident, so no report shares a
seed with a report on the other side. Each class's seeds are shuffled (seed
42) and dealt to distinct folds, so every class keeps at least one seed in
training in every fold.

Commands:
  SIESTA_DEVICE=0 scec_utility.py anonymise     # writes data_scec_utility/aad6c5b/
  scec_utility.py classical                     # LR TF-IDF, XGBoost BoW, SVM TF-IDF
  scec_utility.py export_kit                    # kits/scec_cls/data
"""
import json, os, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
OUT = ROOT / "data_scec_utility" / "aad6c5b"
TECHNIQUES = {"tokenisation": 1, "masking": 2, "substitution": 3, "hashing": 4, "pseudonymisation": 5}
PRESERVED = {"CVE", "MALWARE", "HASH", "TECH", "SOFTWARE", "OPERATING_SYSTEM"}   # diccionarios.DEFAULT_PRESERVED_TYPES at aad6c5b
GOLD_TO_SIESTA = {
    "Person_Name": "PER", "Role_Title": "JOB_TITLE", "Organization": "ORG", "Age": "AGE",
    "Geographic_Location": "LOC", "Country": "LOC", "Facility": "LOC", "IP_Address": "IP",
    "Temporal_IDs": "TIME", "Dates": "DATE", "Software_Information": "SOFTWARE", "Quantity": "QUANTITY",
    "Username": "USERNAME", "Authentication_Credentials": "PASSWORD", "ID_Numbers": "IDENTIFICATION_NUMBER",
    "Email_Address": "EMAIL", "Account_Details": "BANK_OR_CARD_NUMBER", "Vulnerability_Identifiers": "CVE",
    "Operating_System": "OPERATING_SYSTEM", "Currency": "AMOUNT_OF_MONEY", "System_Identifiers": "HOST",
    "Threat_Actor": "ORG", "Malware_Name": "MALWARE", "Phone_Number": "PHONE_NUMBER",
    "Health_Information": "MEDICAL_CONDITION", "Cryptographic_Elements": "HASH",
    "Nationality_Ethnicity": "NATIONALITY", "Event_Information": "MISC", "MAC_Address": "HOST",
    # URL_Domain and File_Resources are split by form in siesta_type()
}


def siesta_type(gold_type, value):
    if gold_type == "URL_Domain":
        return "EMAIL" if "@" in value else ("URL" if "://" in value else "DOMAIN")
    if gold_type == "File_Resources":
        return "FILE_PATH" if ("/" in value or "\\" in value) else "FILE"
    return GOLD_TO_SIESTA[gold_type]


def reports():
    import ast, s_cecilia, splits
    seed = splits.s_cecilia_split()["gold_seed"]
    out = []
    for d in s_cecilia.load():
        spans = d["spans"] if isinstance(d["spans"], list) else ast.literal_eval(d["spans"])
        out.append({"id": d["id"], "text": d["text"], "label": d["cls"], "seed": seed[d["id"]], "spans": spans})
    return out


def gold_entities(doc):
    """Gold spans as SIESTA entities, overlaps resolved longest first."""
    ents, taken = [], []
    for s, e, t in sorted(doc["spans"], key=lambda x: -(x[1] - x[0])):
        if any(s < b and e > a for a, b in taken):
            continue
        taken.append((s, e))
        v = doc["text"][s:e]
        rt = siesta_type(t, v)
        ents.append({"start": s, "end": e, "text": v, "readable_type": rt, "entity_group": rt,
                     "preserved": rt in PRESERVED})
    return sorted(ents, key=lambda x: x["start"])


def anonymise():
    sys.path.insert(0, str(ROOT))
    import cecilia_utility as cu
    a = cu.build_archived("aad6c5b")
    # aad6c5b's shift_date catches ValueError but not OverflowError, so a
    # "date" the shift carries outside the calendar ("20-3-0001", SCEC-GOLD-1129)
    # failed Substitution and Pseudonymisation for the whole report. The fix in
    # the repository (catch both; the value becomes its label) is applied here
    # in-process, and every report it affects is recorded in meta.json.
    import pseudonymise
    overflow = []
    _orig_shift = pseudonymise.shift_date
    def _guarded(value, days):
        try:
            return _orig_shift(value, days)
        except OverflowError:
            overflow.append(value)
            return None
    pseudonymise.shift_date = _guarded
    key = (ROOT / "logs" / "cecilia_utility_key.txt").read_text().strip()
    docs = reports()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "original.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "text": d["text"], "label": d["label"], "seed": d["seed"]}, ensure_ascii=False) + "\n")
    stats = {}
    for tech, opt in TECHNIQUES.items():
        for source in ("gold", "siesta"):
            done = OUT / f"{source}_{tech}.jsonl"
            if done.exists() and sum(1 for _ in open(done)) == len(docs) and (OUT / f"{source}_{tech}.stats.json").exists():
                stats[f"{source}_{tech}"] = json.load(open(OUT / f"{source}_{tech}.stats.json")); continue   # resume
            t0, n_ent, before = time.time(), 0, len(overflow)
            with open(OUT / f"{source}_{tech}.jsonl", "w", encoding="utf-8") as f:
                for d in docs:
                    if source == "siesta":
                        _, ents, mod = a.anonymize(d["text"], opt, hash_key=key, scope="scec-utility")
                        red = [e for e in ents if not e.get("preserved")]
                    else:
                        red = [e for e in gold_entities(d) if not e["preserved"]]
                        if opt == 3:
                            mod = a.replace_with_fake_data(d["text"], red)
                        else:
                            mode = {1: "label", 2: "mask", 4: "hash", 5: "pseudonymise"}[opt]
                            mod = a.replace_entities(d["text"], red, mode=mode, hash_key=key, scope="scec-utility")
                    n_ent += len(red)
                    f.write(json.dumps({"id": d["id"], "text": mod, "label": d["label"], "seed": d["seed"]}, ensure_ascii=False) + "\n")
            stats[f"{source}_{tech}"] = {"seconds": round(time.time() - t0, 1), "redacted_entities": n_ent,
                                         "date_overflow_values": overflow[before:]}
            json.dump(stats[f"{source}_{tech}"], open(OUT / f"{source}_{tech}.stats.json", "w"))
            print(source, tech, stats[f"{source}_{tech}"], flush=True)
    json.dump({"variant": "aad6c5b + date-overflow guard (see anonymise)", "docs": len(docs), "gold_to_siesta": GOLD_TO_SIESTA, "preserved": sorted(PRESERVED),
               "versions": stats}, open(OUT / "meta.json", "w"), indent=1)


def folds():
    docs = reports()
    by_class = defaultdict(set)
    for d in docs:
        by_class[d["label"]].add(d["seed"])
    rng = random.Random(42)
    fold_of, start = {}, 0
    for c in sorted(by_class):
        seeds = sorted(by_class[c]); rng.shuffle(seeds)
        for i, s in enumerate(seeds):
            fold_of[s] = (start + i) % 5
        start += len(seeds)            # rotate so small classes do not all sit in fold 0
    out = []
    for k in range(5):
        te = [i for i, d in enumerate(docs) if fold_of[d["seed"]] == k]
        tr = [i for i, d in enumerate(docs) if fold_of[d["seed"]] != k]
        assert {docs[i]["label"] for i in tr} == set(by_class), f"fold {k}: a class has no training seed"
        out.append({"train": tr, "test": te})
    return out


VERSIONS = ["original"] + [f"{s}_{t}" for s in ("gold", "siesta") for t in TECHNIQUES]


def classical():
    import warnings
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    warnings.filterwarnings("ignore")
    F = folds(); res = []
    for v in VERSIONS:
        docs = [json.loads(l) for l in open(OUT / f"{v}.jsonl", encoding="utf-8")]
        X = np.array([d["text"] for d in docs], dtype=object); labels = sorted({d["label"] for d in docs})
        y = np.array([labels.index(d["label"]) for d in docs])
        for name, make in {"LR TF-IDF": lambda: make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000)),
                           "SVM TF-IDF": lambda: make_pipeline(TfidfVectorizer(min_df=5, ngram_range=(1, 2)), SVC(kernel="linear", class_weight="balanced")),
                           "XGBoost BoW": lambda: make_pipeline(CountVectorizer(min_df=5, ngram_range=(1, 2), max_features=5000), XGBClassifier(eval_metric="mlogloss", random_state=42))}.items():
            for k, f in enumerate(F):
                m = make()
                kw = {"xgbclassifier__sample_weight": compute_sample_weight("balanced", y[f["train"]])} if "XGBoost" in name else {}
                p = m.fit(X[f["train"]], y[f["train"]], **kw).predict(X[f["test"]])
                res.append({"text": v, "model": name, "fold": k, "f1_weighted": f1_score(y[f["test"]], p, average="weighted"),
                            "f1_macro": f1_score(y[f["test"]], p, average="macro"), "accuracy": accuracy_score(y[f["test"]], p)})
            r = [x["f1_weighted"] for x in res if x["text"] == v and x["model"] == name]
            print(f"{v:26} {name:12} F1-w {np.mean(r):.3f} (sd {np.std(r):.3f})", flush=True)
    json.dump(res, open(ROOT / "logs" / "scec_utility_classical_aad6c5b.json", "w"), indent=1)


def export_kit():
    import shutil
    kit = ROOT / "kits" / "scec_cls" / "data"; kit.mkdir(parents=True, exist_ok=True)
    for v in VERSIONS:
        shutil.copy(OUT / f"{v}.jsonl", kit / f"{v}.jsonl")
    shutil.copy(OUT / "meta.json", kit / "anonymisation_meta.json")
    json.dump({"folds": folds()}, open(kit / "split.json", "w"))
    print("kit data:", sorted(p.name for p in kit.iterdir()))


if __name__ == "__main__":
    {"anonymise": anonymise, "classical": classical, "export_kit": export_kit,
     "folds": lambda: print([(len(f["train"]), len(f["test"])) for f in folds()])}[sys.argv[1]]()
