"""Tier B: fine-tune WP17 models on CECILIA, test on CECILIA and S-CECILIA.

Follows WP17/Task 17.2/ner-finetune.py: CECILIA labels collapsed to
CoNLL-03 with its cecilia_to_conll map, IOB2 on every overlapping sub-word
token, 8 epochs, AdamW (weight decay 0.01), batch 16, linear schedule,
best checkpoint by validation loss, seqeval on the held-out 20%.

Deviations, each because the original would be wrong at this data size:
  * warmup 10% of steps -- WP17's 500 warmup steps exceed the whole run
    (~40 steps/epoch), so the learning rate never reaches its peak;
  * lr 2e-5 for large models (5e-5 kept for base), for stability; large
    models also use batch 2 x 8 gradient accumulation (effective 16) and
    gradient checkpointing, to fit an 8 GB GPU;
  * reports longer than 512 tokens are windowed (stride 128) rather than
    truncated, in training and prediction;
  * Auto* classes instead of Bert*, so RoBERTa models load.

Outputs
  cache/ft_<model>/                 best checkpoint
  preds/ft-cecilia__<model>.jsonl   predictions on every S-CECILIA report
  preds_cecilia/ft-cecilia__<model>.jsonl   on CECILIA (score its test split)
  results_finetune.json             seqeval (WP17 protocol) on CECILIA test
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["EVAL_DATASET"] = "cecilia"
import corpora, cecilia, gold  # noqa: E402

CECILIA_TO_CONLL = {  # WP17 ner-finetune.py, verbatim
    "Person": "PER", "Norp": "MISC", "Facilities": "LOC", "GPE_Location": "LOC", "Location": "LOC",
    "Organization": "ORG", "Group": "ORG", "Date": "MISC", "Time": "MISC", "Money": "MISC",
    "Quantities": "MISC", "Numbers_O": "MISC", "Numbers_C": "MISC", "URL": "MISC", "Emails": "MISC",
    "Phone": "MISC", "Address": "LOC", "IP": "MISC"}
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
MAXLEN, STRIDE = 512, 128
dev = torch.device("cuda")


def windows(tok, doc, label2id):
    enc = tok(doc["text"], truncation=True, max_length=MAXLEN, stride=STRIDE,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    out = []
    for w in range(len(enc["input_ids"])):
        offs = enc["offset_mapping"][w]; seq = enc.sequence_ids(w)
        lab = [-100 if seq[i] is None else label2id["O"] for i in range(len(offs))]
        for s, e, l in doc["spans"]:
            t = CECILIA_TO_CONLL.get(l, "MISC"); first = True
            for i, (a, b) in enumerate(offs):
                if seq[i] is None or not (a < e and b > s): continue
                lab[i] = label2id[("B-" if first else "I-") + t]; first = False
        out.append(dict(input_ids=enc["input_ids"][w], attention_mask=enc["attention_mask"][w], labels=lab))
    return out


def batches(items, bs, shuffle, rng, pad_id):
    idx = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for k in range(0, len(idx), bs):
        chunk = [items[i] for i in idx[k:k + bs]]; L = max(len(c["input_ids"]) for c in chunk)
        pad = lambda key, v: torch.tensor([c[key] + [v] * (L - len(c[key])) for c in chunk])
        yield {"input_ids": pad("input_ids", pad_id).to(dev), "attention_mask": pad("attention_mask", 0).to(dev),
               "labels": pad("labels", -100).to(dev)}


def run(model_name):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup, pipeline
    from seqeval.metrics import f1_score, precision_score, recall_score
    torch.manual_seed(42); rng = np.random.default_rng(42)
    large = "large" in model_name
    # Large models: batch 2 x 8 accumulation (same effective 16) with gradient
    # checkpointing, to fit AdamW state for 340M parameters in 8 GB.
    lr, bs, accum = (2e-5, 2, 8) if large else (5e-5, 16, 1)
    out_dir = HERE / "cache" / ("ft_" + model_name.replace("/", "__"))

    docs = {d["id"]: d for d in corpora.load()}; sp = corpora.split()
    tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) if "roberta" in model_name.lower() \
        else AutoTokenizer.from_pretrained(model_name)
    label2id = {l: i for i, l in enumerate(LABELS)}
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label2id,
        ignore_mismatched_sizes=True).to(dev)
    if large: model.gradient_checkpointing_enable()
    tr = [w for i in sp["train"] for w in windows(tok, docs[i], label2id)]
    va = [w for i in sp["val"] for w in windows(tok, docs[i], label2id)]
    te = [w for i in sp["test"] for w in windows(tok, docs[i], label2id)]

    epochs = 8; steps = epochs * -(-len(tr) // (bs * accum))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(); best = float("inf"); t0 = time.time(); log = []
    for ep in range(epochs):
        model.train()
        for k, b in enumerate(batches(tr, bs, True, rng, tok.pad_token_id)):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**b).loss / accum
            scaler.scale(loss).backward()
            if (k + 1) % accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        model.eval(); vl = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in batches(va, bs, False, rng, tok.pad_token_id): vl.append(model(**b).loss.item())
        v = float(np.mean(vl)); log.append(v)
        if v < best:
            best = v; model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
        print(f"{model_name} epoch {ep + 1} val_loss {v:.4f}", flush=True)
    train_secs = time.time() - t0

    # WP17 protocol score: seqeval over sub-word tokens of the test windows
    model = AutoModelForTokenClassification.from_pretrained(out_dir).to(dev).eval()
    yt, yp = [], []
    with torch.no_grad():
        for b in batches(te, bs, False, rng, tok.pad_token_id):
            pr = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.argmax(-1)
            for p, l in zip(pr.tolist(), b["labels"].tolist()):
                yt.append([LABELS[x] for x in l if x != -100])
                yp.append([LABELS[x] for x, m in zip(p, l) if m != -100])
    wp17 = dict(p=precision_score(yt, yp), r=recall_score(yt, yp), f1=f1_score(yt, yp))

    # Span predictions for the common scorer
    tok.model_max_length = MAXLEN
    pipe = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple",
                    stride=STRIDE, device=0)
    meta = {"name": f"{model_name} fine-tuned on CECILIA", "taxonomy": "native", "family": "Tier B: fine-tuned on CECILIA",
            "train_seconds": round(train_secs, 1), "val_loss": log, "lr": lr}
    for corpus, dset in [("scec", gold.load()), ("cecilia", [docs[i] for i in docs])]:
        os.environ["EVAL_DATASET"] = corpus
        with open(corpora.preds_dir() / f"ft-cecilia__{model_name.replace('/', '__')}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for d in dset:
                ents = pipe(d["text"])
                f.write(json.dumps({"id": d["id"], "spans": [[int(e["start"]), int(e["end"]), e["entity_group"]] for e in ents]}) + "\n")
    os.environ["EVAL_DATASET"] = "cecilia"
    res_path = HERE / "results_finetune.json"
    res = json.load(open(res_path)) if res_path.exists() else {}
    res[model_name] = dict(wp17_protocol_cecilia_test=wp17, best_val_loss=best, val_loss=log,
                           train_windows=len(tr), train_seconds=round(train_secs, 1), lr=lr)
    json.dump(res, open(res_path, "w"), indent=1)
    print(model_name, "WP17-protocol CECILIA test", {k: round(v, 3) for k, v in wp17.items()}, flush=True)
    del model, pipe; torch.cuda.empty_cache()


if __name__ == "__main__":
    done = json.load(open(HERE / "results_finetune.json")) if (HERE / "results_finetune.json").exists() else {}
    for m in sys.argv[1:] or [m for m in ["dslim/bert-base-NER", "dbmdz/bert-large-cased-finetuned-conll03-english",
                              "Jean-Baptiste/roberta-large-ner-english"] if m not in done]:
        run(m)
