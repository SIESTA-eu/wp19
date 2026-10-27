#!/bin/bash
# Creates .venv and downloads the RoBERTa-base checkpoint.
# Usage: ./setup.sh [torch index URL]   e.g. https://download.pytorch.org/whl/cu124
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
if [ -n "$1" ]; then pip install torch --index-url "$1"; else pip install torch; fi
pip install -r requirements.txt
python - <<'PY'
import run
from transformers import AutoTokenizer, AutoModelForSequenceClassification
m = run.MODELS["roberta"]
AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m, num_labels=6); print("ok", m)
import torch; print("CUDA available:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
PY
echo "Setup done."
