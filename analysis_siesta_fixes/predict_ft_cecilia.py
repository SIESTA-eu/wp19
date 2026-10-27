"""Predictions of the three WP17 models fine-tuned on CECILIA (finetune.py
checkpoints in cache/ft_*) on corpora finetune.py never predicted on.
Same pipeline settings as finetune.py: aggregation "simple", 512-token windows, stride 128.

Run from WP17/eval:  .venv/bin/python analysis_siesta_fixes/predict_ft_cecilia.py tab cyberpii casie
"""
import json, os, sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL)); os.chdir(EVAL)
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODELS = ["dslim/bert-base-NER", "dbmdz/bert-large-cased-finetuned-conll03-english", "Jean-Baptiste/roberta-large-ner-english"]

for corpus in sys.argv[1:]:
    os.environ["EVAL_DATASET"] = corpus
    import importlib, corpora
    importlib.reload(corpora)
    docs = corpora.load()
    for m in MODELS:
        d = EVAL / "cache" / f"ft_{m.replace('/', '__')}"
        tok = AutoTokenizer.from_pretrained(d); tok.model_max_length = 512
        pipe = pipeline("token-classification", model=AutoModelForTokenClassification.from_pretrained(d), tokenizer=tok,
                        aggregation_strategy="simple", stride=128, device=0)
        out = corpora.preds_dir() / f"ft-cecilia__{m.replace('/', '__')}.jsonl"
        meta = {"name": f"{m} fine-tuned on CECILIA", "taxonomy": "native", "family": "Tier B: fine-tuned on CECILIA",
                "note": "checkpoint from finetune.py, predicted by analysis_siesta_fixes/predict_ft_cecilia.py"}
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for x in docs:
                f.write(json.dumps({"id": x["id"], "spans": [[int(e["start"]), int(e["end"]), e["entity_group"]] for e in pipe(x["text"])]}) + "\n")
        print(corpus, "wrote", out.name, flush=True)
