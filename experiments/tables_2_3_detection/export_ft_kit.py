"""Write the data of d17_on_scecilia_kit/: WP17 models fine-tuned on S-CECILIA, run on a
server GPU (Tier C for the WP17 models: trained on the training seeds, tested on
the held-out ones).

  data/train.jsonl, val.jsonl   S-CECILIA training-seed reports WITH gold spans
                                (training needs them); val = all reports of
                                VAL_SEEDS training seeds, chosen at random
                                (seed 42), so validation is seed-grouped too
  data/predict_<corpus>.jsonl   texts only, no gold: scec (247 held-out),
                                cecilia (all 879; score its test split here),
                                tab (127), cyner (94)
  data/labels.json              siesta: SCEC_TO_SIESTA, for RubiNER's general NER;
                                conll4: WP17's collapse (score.SCEC_TO_CONLL,
                                anything else MISC); native: the 31 S-CECILIA types
  data/MANIFEST.json            counts and SHA-256 of each file

No held-out gold leaves this machine. Usage: export_ft_kit.py
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
KIT = ROOT / "experiments" / "tables_2_3_detection" / "d17_on_scecilia_kit" / "data"
VAL_SEEDS = 4
# S-CECILIA types as SIESTA entity types (D19.4, Appendix 1), the labels of
# RubiNER's general NER; None = no SIESTA equivalent, trained as O.
SCEC_TO_SIESTA = {
    "Person_Name": "PER", "Organization": "ORG", "Geographic_Location": "LOC", "Country": "LOC",
    "Facility": "LOC", "Dates": "DATE", "Temporal_IDs": "TIME", "Role_Title": "JOB_TITLE", "Age": "AGE",
    "Nationality_Ethnicity": "NATIONALITY", "Username": "USERNAME", "Phone_Number": "PHONE_NUMBER",
    "Email_Address": "EMAIL", "Quantity": "QUANTITY", "Authentication_Credentials": "PASSWORD",
    "ID_Numbers": "IDENTIFICATION_NUMBER", "Account_Details": "BANK_OR_CARD_NUMBER",
    "Health_Information": "MEDICAL_CONDITION", "Currency": "AMOUNT_OF_MONEY", "IP_Address": "IP",
    "URL_Domain": "URL", "File_Resources": "FILE", "Cryptographic_Elements": "HASH",
    "Software_Information": "SOFTWARE", "Operating_System": "OPERATING_SYSTEM", "Malware_Name": "MALWARE",
    "Threat_Actor": "MALWARE", "Vulnerability_Identifiers": "CVE",
    "MAC_Address": None, "System_Identifiers": None, "Event_Information": None}


def main():
    import corpora, score
    KIT.mkdir(parents=True, exist_ok=True)
    sp = json.load(open(ROOT / "corpus_loaders" / "split.json"))
    os.environ["EVAL_DATASET"] = "scec"
    docs = {d["id"]: d for d in corpora.load()}
    train_ids = sorted(sp["train"])
    seeds = sorted({sp["gold_seed"][i] for i in train_ids})
    assert not set(seeds) & set(sp["test_seeds"])
    val_seeds = sorted(np.random.default_rng(42).choice(seeds, VAL_SEEDS, replace=False).tolist())
    manifest = {"train_seeds": len(seeds), "val_seeds": val_seeds}
    for name, ids in (("train", [i for i in train_ids if sp["gold_seed"][i] not in val_seeds]),
                      ("val", [i for i in train_ids if sp["gold_seed"][i] in val_seeds])):
        with open(KIT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps({"id": i, "seed": sp["gold_seed"][i], "text": docs[i]["text"],
                                    "spans": [list(s) for s in docs[i]["spans"]]}, ensure_ascii=False) + "\n")
        manifest[name] = len(ids)
    native = sorted({l for d in docs.values() for _, _, l in d["spans"]})
    json.dump({"conll4": {l: score.SCEC_TO_CONLL.get(l, "MISC") for l in native},
               "native": {l: l for l in native},
               "siesta": {l: SCEC_TO_SIESTA[l] for l in native}}, open(KIT / "labels.json", "w"), indent=1)
    for corpus in ("scec", "cecilia", "tab", "cyner", "casie", "cyberpii"):
        os.environ["EVAL_DATASET"] = corpus
        cd = corpora.load()
        if corpus == "scec":
            test = set(sp["test"]); cd = [d for d in cd if d["id"] in test]
        with open(KIT / f"predict_{corpus}.jsonl", "w", encoding="utf-8") as f:
            for d in cd:
                f.write(json.dumps({"id": d["id"], "text": d["text"]}, ensure_ascii=False) + "\n")
        manifest[f"predict_{corpus}"] = len(cd)
    for p in sorted(KIT.iterdir()):
        if p.name != "MANIFEST.json":
            manifest[f"sha256:{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    json.dump(manifest, open(KIT / "MANIFEST.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if not k.startswith("sha256")}))


if __name__ == "__main__":
    main()
