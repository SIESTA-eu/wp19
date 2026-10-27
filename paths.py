"""Repository layout. Every script imports this first: it puts each code folder
on sys.path, so modules import one another by name wherever they live, and
ROOT anchors data, predictions and outputs at the repository root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ["rubiner", "rubiner/thresholds", "corpus_loaders", "scoring",
        "experiments/tables_2_3_detection", "experiments/table_4_classification",
        "experiments/table_5_spy"]
for d in reversed(CODE):
    if str(ROOT / d) not in sys.path:
        sys.path.insert(0, str(ROOT / d))
