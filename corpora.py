"""Which corpus a runner works on, chosen by EVAL_DATASET (default scec).

  scec     S-CECILIA gold (974 reports), split.json (seed-grouped)
  anonlfi_cases  AnonLFI 2.0's two published case studies (anonlfi_cases.py)
  tab      Text Anonymization Benchmark, ECHR test split (benchmarks.py)
  cyner    CyNER MITRE test split (benchmarks.py)
  casie    CASIE news articles on cyber incidents, event arguments (casie.py; recall only)
  cyberpii CyberPII-Bench, corrected gold (cyberpii.py, CYBERPII_CORRECTIONS.md)
  cecilia  CECILIA-10C-900-NER (WP17 / D17.1), split_cecilia.json: WP17's
           own protocol, sklearn train_test_split(test_size=0.2,
           random_state=42), then 1/8 of the rest for validation
"""
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def name():
    return os.environ.get("EVAL_DATASET", "scec")


def load():
    if name() in ("tab", "cyner"):
        import benchmarks
        return getattr(benchmarks, name())()
    if name() == "anonlfi_cases":
        import anonlfi_cases
        return anonlfi_cases.load()
    if name() in ("casie", "cyberpii"):
        import importlib
        return importlib.import_module(name()).load()
    if name() == "cecilia":
        import cecilia
        return cecilia.load()[0]
    import gold
    return gold.load()


def preds_dir():
    d = HERE / ("preds" if name() == "scec" else f"preds_{name()}")
    d.mkdir(exist_ok=True)
    return d


def gold_taxonomy():
    return {"scec": "scec", "cecilia": "cecilia"}.get(name(), "native")


def split():
    if name() == "cecilia":
        p = HERE / "split_cecilia.json"
        if not p.exists():
            from sklearn.model_selection import train_test_split
            ids = [d["id"] for d in load()]
            rest, test = train_test_split(ids, test_size=0.2, random_state=42)
            train, val = train_test_split(rest, test_size=0.1 / 0.8, random_state=42)
            json.dump(dict(train=train, val=val, test=test), open(p, "w"), indent=1)
        return json.load(open(p))
    return json.load(open(HERE / "split.json"))
