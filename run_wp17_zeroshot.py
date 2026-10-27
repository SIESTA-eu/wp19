"""The 14 WP17 (D17.1 Table 2) models, off the shelf, on every gold report.

HF models go through the token-classification pipeline with a 128-token
stride, so reports longer than 512 tokens are read in full rather than
truncated. spaCy models use their own NER.
"""
import json, sys, time
from pathlib import Path
import torch
import corpora

HERE = Path(__file__).resolve().parent
HF = ["dslim/bert-base-NER", "dslim/bert-large-NER",
      "dbmdz/bert-base-cased-finetuned-conll03-english", "dbmdz/bert-large-cased-finetuned-conll03-english",
      "elastic/distilbert-base-uncased-finetuned-conll03-english", "Davlan/xlm-roberta-base-ner-hrl",
      "FacebookAI/xlm-roberta-large-finetuned-conll03-english", "Jean-Baptiste/roberta-large-ner-english",
      "Davlan/xlm-roberta-large-ner-hrl", "Davlan/bert-base-multilingual-cased-ner-hrl"]
SPACY = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]


def write(name, rows, extra):
    out = corpora.preds_dir() / (name.replace("/", "__") + ".jsonl")
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"name": name, "taxonomy": "native", "family": "WP17 zero-shot", **extra}}) + "\n")
        for gid, spans in rows: f.write(json.dumps({"id": gid, "spans": spans}) + "\n")


# dbmdz/bert-base-cased-finetuned-conll03-english ships with LABEL_0..8 in
# its config, so the pipeline cannot tell O from an entity. These are the
# CoNLL-2003 names in the order its large sibling declares; the base
# model's outputs fit it (LABEL_0 dominates, names land on LABEL_4 = I-PER).
CONLL03_ORDER = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def run_hf(model, docs):
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model)
    tok.model_max_length = 512
    cfg = AutoConfig.from_pretrained(model)
    if all(v.startswith("LABEL_") for v in cfg.id2label.values()) and len(cfg.id2label) == 9:
        cfg.id2label = dict(enumerate(CONLL03_ORDER)); cfg.label2id = {v: k for k, v in cfg.id2label.items()}
    mdl = AutoModelForTokenClassification.from_pretrained(model, config=cfg)
    pipe = pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple",
                    stride=128, device=0 if torch.cuda.is_available() else -1)
    t0, rows = time.time(), []
    for d in docs:
        ents = pipe(d["text"])
        rows.append((d["id"], [[int(e["start"]), int(e["end"]), e["entity_group"]] for e in ents]))
    return rows, time.time() - t0


def run_spacy(model, docs):
    import spacy
    if model.endswith("trf"): spacy.prefer_gpu()
    nlp = spacy.load(model)
    t0, rows = time.time(), []
    for d, doc in zip(docs, nlp.pipe([d["text"] for d in docs], batch_size=16)):
        rows.append((d["id"], [[e.start_char, e.end_char, e.label_] for e in doc.ents]))
    return rows, time.time() - t0


if __name__ == "__main__":
    docs = corpora.load()
    only = sys.argv[1:]
    for m in HF + SPACY:
        if only and m not in only: continue
        name = "spacy/" + m if m in SPACY else m
        if (corpora.preds_dir() / (name.replace("/", "__") + ".jsonl")).exists(): continue
        rows, secs = (run_spacy if m in SPACY else run_hf)(m, docs)
        write(name, rows, {"seconds": round(secs, 1), "device": "cuda"})
        print(f"{m}: {secs:.0f}s", flush=True)
        torch.cuda.empty_cache()
