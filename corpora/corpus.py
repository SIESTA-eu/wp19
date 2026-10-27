"""The corpus a script works on, chosen by the EVAL_DATASET environment
variable (default scec):

  scec     S-CECILIA-Gold, 974 reports (s_cecilia.py)
  cecilia  CECILIA-10C-900, as used by D17 (cecilia.py)
  cyner    CyNER, MITRE test split (cyner.py)

Every loader returns a list of documents: dict(id, text, spans), spans being
(start, end, label) character offsets.
"""
import os
from pathlib import Path

import cecilia, cyner, s_cecilia, splits

ROOT = Path(__file__).resolve().parents[1]
LOADERS = {"scec": s_cecilia.load, "cecilia": lambda: cecilia.load()[0], "cyner": cyner.load}
SPLITS = {"scec": splits.s_cecilia_split, "cecilia": splits.cecilia_split}
# Label set of each corpus's gold, for score.py; "native" = scored type-agnostically only.
TAXONOMY = {"scec": "scec", "cecilia": "cecilia", "cyner": "native"}


def name():
    return os.environ.get("EVAL_DATASET", "scec")


def load():
    return LOADERS[name()]()


def split():
    """Train/test split of the corpus (S-CECILIA and CECILIA only; CyNER is all test)."""
    return SPLITS[name()]()


def gold_taxonomy():
    return TAXONOMY[name()]


def preds_dir():
    """Where predictions on this corpus are written: preds/ for S-CECILIA, else preds_<name>/."""
    d = ROOT / ("preds" if name() == "scec" else f"preds_{name()}")
    d.mkdir(exist_ok=True)
    return d
