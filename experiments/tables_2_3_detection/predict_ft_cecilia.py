"""Predictions of the three WP17 models fine-tuned on CECILIA (finetune.py
checkpoints in cache/ft_*) on corpora finetune.py never predicted on.
Same pipeline settings as finetune.py: aggregation "simple", 512-token windows, stride 128.

Run from the repository root:  .venv/bin/python experiments/tables_2_3_detection/predict_ft_cecilia.py cyner
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
os.chdir(ROOT)
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODELS = ["dslim/bert-base-NER", "dbmdz/bert-large-cased-finetuned-conll03-english", "Jean-Baptiste/roberta-large-ner-english"]

for cname in sys.argv[1:]:
    os.environ["EVAL_DATASET"] = cname
    import importlib, corpus
    importlib.reload(corpus)
    docs = corpus.load()
    for m in MODELS:
        d = ROOT / "cache" / f"ft_{m.replace('/', '__')}"
        tok = AutoTokenizer.from_pretrained(d); tok.model_max_length = 512
        pipe = pipeline("token-classification", model=AutoModelForTokenClassification.from_pretrained(d), tokenizer=tok,
                        aggregation_strategy="simple", stride=128, device=0)
        out = corpus.preds_dir() / f"ft-cecilia__{m.replace('/', '__')}.jsonl"
        meta = {"name": f"{m} fine-tuned on CECILIA", "taxonomy": "native", "family": "Tier B: fine-tuned on CECILIA",
                "note": "checkpoint from finetune.py, predicted by predict_ft_cecilia.py"}
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for x in docs:
                f.write(json.dumps({"id": x["id"], "spans": [[int(e["start"]), int(e["end"]), e["entity_group"]] for e in pipe(x["text"])]}) + "\n")
        print(cname, "wrote", out.name, flush=True)
