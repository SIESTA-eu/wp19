"""Incident-classification utility of RubiNER-anonymised text, with the existing
utility code unchanged: cecilia_utility.py (CECILIA-10C-900, 5-fold, the CECILIA
paper's protocol) and haughton_compare.py (Haughton et al.'s 6-class split).
Both build SIESTA through cecilia_utility.build_archived(); here it returns RubiNER
instead, so the same key, folds, techniques and classifiers apply.

  anonymise   data_cecilia_utility/rubiner/ and data_haughton/rubiner/
  classical   cecilia_utility baselines (LR TF-IDF, k-NN BoW; 10C and 6C) and
              haughton_compare classical (the paper's ten classical pairs, 6C split)
  kit         kits/cecilia_cls_rubiner/data: the text versions and the same folds,
              for the transformer runs (kits/cecilia_cls/run.py, unchanged)

Run from the repository root:
  SIESTA_DEVICE=0 .venv/bin/python experiments/table_4_classification/utility_rubiner.py anonymise
"""
import json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
VARIANT = "rubiner"


def build_rubiner(_variant=None):
    import run_siesta, custom_entities  # run_siesta puts SIESTA_REPO (the corrected copy) on the path
    import run_rubiner as ft
    from base_anonymizer import windowed_ner_pipeline
    import torch
    custom_entities._load()
    if run_siesta.DEV >= 0 and custom_entities._model is not None:
        custom_entities._model = custom_entities._model.to("cuda")
    a = run_siesta.build("aad-fixC")
    a.pipelines.pop("general")
    pipe = windowed_ner_pipeline(str(ft.checkpoint("siesta")))
    if run_siesta.DEV >= 0:
        pipe.model.to("cuda"); pipe.device = torch.device("cuda")
    base, extra = a.detect_ner, ft.ft_detector(pipe)
    a.detect_ner = lambda text: base(text) + extra(text)
    os.chdir(ROOT)
    return a


def anonymise():
    import cecilia_utility as cu, haughton_compare as hc
    a = build_rubiner()
    cu.build_archived = lambda variant: a
    cu.ARCHIVES[VARIANT] = None          # makes cu.anonymise take the build_archived path
    cu.anonymise(VARIANT)
    hc.anonymise(VARIANT)


def classical():
    import cecilia_utility as cu, haughton_compare as hc
    cu.baselines(VARIANT)
    hc.classical(VARIANT)


def kit():
    import numpy as np, cecilia_utility as cu
    src, dst = ROOT / "kits" / "cecilia_cls", ROOT / "kits" / "cecilia_cls_rubiner"
    if not dst.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("data", "results*", "work", ".venv", "logs"))
    data = dst / "data"; data.mkdir(exist_ok=True)
    for tech in ["original"] + list(cu.TECHNIQUES):
        shutil.copy(cu.OUT / VARIANT / f"{tech}.jsonl", data / f"{tech}.jsonl")
    shutil.copy(cu.OUT / VARIANT / "meta.json", data / "anonymisation_meta.json")
    docs = [json.loads(l) for l in open(cu.OUT / VARIANT / "original.jsonl", encoding="utf-8")]
    folds = {s: [{"train": tr.tolist(), "test": te.tolist()} for tr, te in cu.folds(np.array([d[f"label_{s}"] for d in docs]))]
             for s in ("10c", "6c")}
    json.dump(folds, open(data / "folds.json", "w"))
    same = json.load(open(src / "data" / "folds.json")) == folds
    print("kit data:", sorted(p.name for p in data.iterdir()), "| folds identical to the release kit:", same)


if __name__ == "__main__":
    {"anonymise": anonymise, "classical": classical, "kit": kit}[sys.argv[1]]()
