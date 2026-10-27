"""Transformer classifiers of Haughton et al. (2026, NLPAICS, "When Privacy
Helps...") on the same CECILIA incidents: original text, the paper's manual
pseudonymisation (labels, xxxx, fake data) and SIESTA's automatic
anonymisation (five techniques), all on one split.

Refinement run (21 Oct 2026): RoBERTa-base at the paper's batch 8. The earlier
SecureBERT 2.0 attempt (results/securebert2, stopped) used batch 4 x 2
accumulation steps on the 8 GB GPU.

Settings as in the notebook that produced the paper's results
(Transformers_Clean (2).ipynb) and the paper's Section 4.1: Hugging Face
Trainer, 5 epochs, batch 8, learning rate 5e-5, warm-up 10% of steps, weight
decay 0.01, unweighted loss, inputs truncated to 512 tokens (or the model's
own maximum if smaller); the paper's eleven models; 70/30 stratified split,
seed 42, on the 6-class subset (the four rarest classes dropped).

Model selection: the notebook evaluates on the test set after every epoch and
keeps the epoch with the lowest test loss (load_best_model_at_end,
metric_for_best_model="eval_loss"). Each job here records the test
predictions and test loss after every epoch, so both are available from one
run: "best" (the paper's rule, to reproduce its figures) and "last" (the model
after epoch 5, no selection on the test set).

The paper trains each model once (seed 42); every job here is repeated with
five training seeds (42-46) on the same split.

One job = model x scheme x text version x seed; each writes
results/<model>/<scheme>/<text>/seed<k>.json (metrics, per-epoch predictions,
and under "params" and "environment" every training argument, the model and
tokenizer revision, data and split hashes, library versions and GPU);
finished jobs are skipped.

Usage:
  python run.py --list
  python run.py --shard 0/8
  python run.py --models roberta --texts paper_labels --seeds 42 --smoke
"""
import argparse, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
MODELS = {"roberta": "roberta-base", "bert": "bert-base-uncased", "secbert": "jackaduma/SecBERT",
          "secroberta": "jackaduma/SecRoBERTa", "distilbert": "distilbert-base-uncased", "albert": "albert-base-v2",
          "mpnet": "sentence-transformers/all-mpnet-base-v2", "deberta": "microsoft/deberta-v3-base",
          "securebert": "ehsanaghaei/SecureBERT", "securebert2": "cisco-ai/SecureBERT2.0-base",
          "cybertuned": "s2w-ai/CyBERTuned-SecurityLLM"}
TEXTS = ['original', 'manual_labels', 'manual_masking', 'rubiner_tokenisation', 'rubiner_masking', 'rubiner_pseudonymisation', 'wp17_dslim_tokenisation', 'wp17_dslim_masking', 'wp17_roberta_tokenisation', 'wp17_roberta_masking']
SCHEMES = ["6c", "10c"]
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 5
SMOKE = False


def metrics(y, p):
    from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
    out = {"accuracy": accuracy_score(y, p), "mcc": matthews_corrcoef(y, p)}
    for avg in ("weighted", "macro"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=avg, zero_division=0)
        out.update({f"prec_{avg}": pr, f"recall_{avg}": rc, f"f1_{avg}": f1})
    return out


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    """Versions and hardware, recorded with every result."""
    import platform, sys, torch, transformers, tokenizers, sklearn, numpy
    try:
        import accelerate; acc = accelerate.__version__
    except ImportError:
        acc = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "accelerate": acc, "scikit_learn": sklearn.__version__, "numpy": numpy.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count()}


def run_job(model, scheme, text, seed):
    import numpy as np, torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)
    out = (HERE / "results_smoke" if SMOKE else RESULTS) / model / scheme / text / f"seed{seed}.json"
    if out.exists():
        return "skipped"
    set_seed(seed)
    docs = [json.loads(l) for l in open(DATA / f"{text}.jsonl", encoding="utf-8")]
    split = json.load(open(DATA / "split.json"))[scheme]
    labels = sorted({docs[i]["label"] for i in split["train"] + split["test"]})
    lid = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(MODELS[model])
    model_max = getattr(tok, "model_max_length", 512)
    max_len = min(512, model_max if model_max <= 10000 else 512)   # as the notebook

    class DS(torch.utils.data.Dataset):
        def __init__(self, idx):
            self.enc = tok([docs[i]["text"] for i in idx], truncation=True, max_length=max_len)
            self.y = [lid[docs[i]["label"]] for i in idx]
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.y[i]}

    tr, te = DS(split["train"][:32] if SMOKE else split["train"]), DS(split["test"])
    mdl = AutoModelForSequenceClassification.from_pretrained(MODELS[model], num_labels=len(labels),
                                                             ignore_mismatched_sizes=True)
    per_epoch = []

    def keep(ev):
        logits = ev.predictions[0] if isinstance(ev.predictions, tuple) else ev.predictions
        per_epoch.append([labels[i] for i in np.argmax(logits, axis=-1)])
        return {}

    work = HERE / "work" / f"{model}_{scheme}_{text}_{seed}"
    targs = TrainingArguments(output_dir=str(work), num_train_epochs=1 if SMOKE else EPOCHS, learning_rate=5e-5,
                              per_device_train_batch_size=8, per_device_eval_batch_size=8, warmup_ratio=0.1,
                              weight_decay=0.01, seed=seed, eval_strategy="epoch", save_strategy="no",
                              logging_steps=10, report_to=[])
    t0 = time.time()
    trainer = Trainer(model=mdl, args=targs, train_dataset=tr, eval_dataset=te,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=keep)
    trainer.train()
    losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    y = [labels[i] for i in te.y]
    best = int(np.argmin(losses))
    rec = {"model": model, "checkpoint": MODELS[model], "scheme": scheme, "text": text, "seed": seed,
           "max_len": max_len, "seconds": round(time.time() - t0, 1),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "eval_loss": losses, "best_epoch": best + 1,
           "metrics_last": metrics(y, per_epoch[-1]), "metrics_best": metrics(y, per_epoch[best]),
           "metrics_by_epoch": [metrics(y, p) for p in per_epoch],
           "test_ids": [docs[i]["id"] for i in split["test"]], "pred_by_epoch": per_epoch, "gold": y,
           "params": {
               "training_arguments": json.loads(trainer.args.to_json_string()),
               "model_checkpoint": MODELS[model],
               "model_revision": getattr(mdl.config, "_commit_hash", None),
               "tokenizer_class": type(tok).__name__, "tokenizer_model_max_length": model_max,
               "max_len": max_len, "truncation": True, "padding": "dynamic (DataCollatorWithPadding)",
               "loss": "cross-entropy, unweighted", "num_labels": len(labels), "labels": labels,
               "train_size": len(tr), "test_size": len(te), "smoke": SMOKE,
               "selection_rules": {"last": f"model after epoch {targs.num_train_epochs}",
                                   "best": "epoch with the lowest test loss (paper's load_best_model_at_end on eval_loss)"},
               "data_file": f"data/{text}.jsonl", "data_sha256": _sha256(DATA / f"{text}.jsonl"),
               "split_file": "data/split.json", "split_sha256": _sha256(DATA / "split.json"),
               "run_py_sha256": _sha256(__file__)},
           "environment": environment()}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part"); tmp.write_text(json.dumps(rec)); tmp.rename(out)
    import shutil; shutil.rmtree(work, ignore_errors=True)
    return (f"F1-w last {rec['metrics_last']['f1_weighted']:.4f}, best (epoch {best + 1}) "
            f"{rec['metrics_best']['f1_weighted']:.4f} in {rec['seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    # Default: RoBERTa-base, 6C, ten text versions, five seeds: 50 jobs.
    ap.add_argument("--models", nargs="+", default=["roberta"], choices=list(MODELS))
    ap.add_argument("--schemes", nargs="+", default=["6c"], choices=SCHEMES)
    ap.add_argument("--texts", nargs="+", default=TEXTS, choices=TEXTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="32 training documents, 1 epoch, results_smoke/")
    args = ap.parse_args()
    global SMOKE
    SMOKE = args.smoke
    k, n = map(int, args.shard.split("/"))
    # Seed-major order: every model and text gets seed 42 first, so a run cut
    # short still has one complete seed (the paper's) for everything.
    todo = [(m, s, t, e) for e in args.seeds for m in args.models for s in args.schemes for t in args.texts][k::n]
    if args.list:
        print("\n".join(" ".join(map(str, j)) for j in todo)); print(len(todo), "jobs"); return
    for i, j in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {' '.join(map(str, j))}: {run_job(*j)}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
