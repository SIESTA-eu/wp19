"""Fine-tune WP17 models on S-CECILIA's training seeds; predict on the held-out
seeds and three other corpora. Server-GPU kit; scoring is done on the project
machine (no held-out gold is in this kit).

Training is WP17/eval/finetune.py's (the WP17 recipe, as used for the models
fine-tuned on CECILIA): 8 epochs, AdamW (weight decay 0.01), effective batch
16, linear schedule with 10% warmup, lr 5e-5 (base) / 2e-5 (large, batch
2 x 8 with gradient checkpointing), fp16 autocast, IOB2 on every overlapping
sub-word token, 512-token windows with stride 128, best epoch by validation
loss, seed 42. Kept identical on purpose, although a large GPU would allow
bigger batches. Only the data differs:
  train/val   data/train.jsonl, data/val.jsonl (validation = 4 whole training
              seeds, so it is seed-grouped like the test set)
  --labels    conll4: S-CECILIA types collapsed to PER/ORG/LOC/MISC as in
              WP17's recipe; native: the 31 S-CECILIA types

Writes results/<model>__<labels>/: train_log.json, env.json and
<corpus>.jsonl (one line per document: id and predicted spans). Resumable:
a finished training (checkpoint/ + train_log.json) is not repeated, and a
corpus whose file is complete is skipped.

Usage:
  python run.py --model dslim/bert-base-NER --labels conll4 [--gpu 0]
  python run.py --model dslim/bert-base-NER --labels conll4 --smoke   # test
"""
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
MAXLEN, STRIDE = 512, 128
CORPORA = ("scec",)  # local copy: SIESTA itself is run on every corpus, so only the held-out seeds are predicted here


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


# --- from WP17/eval/finetune.py (label map passed in) ----------------------
def windows(tok, doc, label2id, to_label):
    enc = tok(doc["text"], truncation=True, max_length=MAXLEN, stride=STRIDE,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    out = []
    for w in range(len(enc["input_ids"])):
        offs = enc["offset_mapping"][w]; seq = enc.sequence_ids(w)
        lab = [-100 if seq[i] is None else label2id["O"] for i in range(len(offs))]
        for s, e, l in doc["spans"]:
            t = to_label[l]; first = True
            if t is None: continue  # "siesta" label set: type with no SIESTA equivalent, left as O
            for i, (a, b) in enumerate(offs):
                if seq[i] is None or not (a < e and b > s): continue
                lab[i] = label2id[("B-" if first else "I-") + t]; first = False
        out.append(dict(input_ids=enc["input_ids"][w], attention_mask=enc["attention_mask"][w], labels=lab))
    return out


def batches(items, bs, shuffle, rng, pad_id, dev):
    import numpy as np, torch
    idx = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for k in range(0, len(idx), bs):
        chunk = [items[i] for i in idx[k:k + bs]]; L = max(len(c["input_ids"]) for c in chunk)
        pad = lambda key, v: torch.tensor([c[key] + [v] * (L - len(c[key])) for c in chunk])
        yield {"input_ids": pad("input_ids", pad_id).to(dev), "attention_mask": pad("attention_mask", 0).to(dev),
               "labels": pad("labels", -100).to(dev)}
# ---------------------------------------------------------------------------


def env_info():
    import torch, transformers
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        gpu = None
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__, device=torch.cuda.get_device_name(0), nvidia_smi=gpu)


def train(model_name, labelset, out, smoke):
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
    dev = torch.device("cuda")
    torch.manual_seed(42); rng = np.random.default_rng(42)
    to_label = json.load(open(DATA / "labels.json"))[labelset]
    types = sorted({v for v in to_label.values() if v})
    LABELS = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]
    label2id = {l: i for i, l in enumerate(LABELS)}
    large = "large" in model_name
    lr, bs, accum = (2e-5, 2, 8) if large else (5e-5, 16, 1)
    ck = out / "checkpoint"
    tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) if "roberta" in model_name.lower() \
        else AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label2id,
        ignore_mismatched_sizes=True).to(dev)
    if large: model.gradient_checkpointing_enable()
    trd, vad = jl(DATA / "train.jsonl"), jl(DATA / "val.jsonl")
    if smoke: trd, vad = trd[:20], vad[:5]
    tr = [w for d in trd for w in windows(tok, d, label2id, to_label)]
    va = [w for d in vad for w in windows(tok, d, label2id, to_label)]
    epochs = 1 if smoke else 8; steps = epochs * -(-len(tr) // (bs * accum))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(); best = float("inf"); t0 = time.time(); log = []
    for ep in range(epochs):
        model.train()
        for k, b in enumerate(batches(tr, bs, True, rng, tok.pad_token_id, dev)):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**b).loss / accum
            scaler.scale(loss).backward()
            if (k + 1) % accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        model.eval(); vl = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in batches(va, bs, False, rng, tok.pad_token_id, dev): vl.append(model(**b).loss.item())
        v = float(np.mean(vl)); log.append(v)
        if v < best:
            best = v; model.save_pretrained(ck); tok.save_pretrained(ck)
        print(f"{model_name} {labelset} epoch {ep + 1} val_loss {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(model=model_name, labels=labelset, label_list=LABELS, lr=lr, batch=bs, accum=accum,
                   epochs=epochs, train_docs=len(trd), train_windows=len(tr), val_docs=len(vad),
                   val_loss=log, best_val_loss=best, best_epoch=int(np.argmin(log)) + 1,
                   train_seconds=round(time.time() - t0, 1), smoke=smoke),
              open(out / "train_log.json", "w"), indent=1)
    del model; torch.cuda.empty_cache()


def predict(out, smoke):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    ck = out / "checkpoint"
    tok = AutoTokenizer.from_pretrained(ck); tok.model_max_length = MAXLEN
    model = AutoModelForTokenClassification.from_pretrained(ck).to("cuda").eval()
    pipe = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple",
                    stride=STRIDE, device=0)
    for corpus in CORPORA:
        docs = jl(DATA / f"predict_{corpus}.jsonl")
        if smoke: docs = docs[:3]
        dst = out / f"{corpus}.jsonl"
        if dst.exists() and sum(1 for _ in open(dst)) == len(docs):
            print(f"{out.name} {corpus}: done, skipped", flush=True); continue
        t0 = time.time()
        with open(dst.with_suffix(".part"), "w", encoding="utf-8") as f:
            for d in docs:
                ents = pipe(d["text"]) if d["text"].strip() else []
                f.write(json.dumps({"id": d["id"], "spans": [[int(e["start"]), int(e["end"]), e["entity_group"]]
                                                             for e in ents]}) + "\n")
        dst.with_suffix(".part").rename(dst)
        print(f"{out.name} {corpus}: {len(docs)} documents ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", choices=("conll4", "native", "siesta"), required=True)
    ap.add_argument("--gpu", type=int, default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--smoke", action="store_true", help="20 training reports, 1 epoch, 3 documents per corpus")
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    out = RESULTS / (("SMOKE__" if a.smoke else "") + f"{a.model.replace('/', '__')}__{a.labels}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(env_info(), open(out / "env.json", "w"), indent=1)
    if (out / "train_log.json").exists() and (out / "checkpoint").exists():
        print(f"{out.name}: trained, skipping training", flush=True)
    else:
        train(a.model, a.labels, out, a.smoke)
    predict(out, a.smoke)
    print(f"{out.name} ALL DONE", flush=True)


if __name__ == "__main__":
    main()
