"""Turn the results of kits/ft_scec (WP17 models fine-tuned on the S-CECILIA
training seeds, run on a server GPU) into prediction files for score.py.

Reads results/<model>__<labels>/<corpus>.jsonl, env.json and train_log.json
from an unpacked results directory or a .tar.gz, and writes
preds[_<corpus>]/ft_scec__<model>__<labels>.jsonl with a _meta line:
  labels native  the 31 S-CECILIA types -> taxonomy "scec" (score.py then
                 also computes the strict 31-label F1 on S-CECILIA)
  labels conll4  PER / ORG / LOC / MISC (WP17's collapse) -> taxonomy "native"
A corpus with documents missing is reported and skipped. CECILIA is
predicted in full (879 reports); score.py scores its test split.

Usage: import_ft_kit.py <results dir or ft_scec_results.tar.gz>
"""
import json, os, shutil, sys, tarfile, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = {"scec": 247, "cecilia": 879, "tab": 127, "cyner": 94, "casie": 1000, "cyberpii": 78}


def main(src):
    src = Path(src)
    tmp = None
    if src.suffix == ".gz":
        tmp = tempfile.mkdtemp(dir=HERE / "logs")
        with tarfile.open(src) as t:
            t.extractall(tmp, filter="data")
        src = Path(tmp)
    sys.path.insert(0, str(HERE))
    dirs = sorted(p.parent for p in src.rglob("train_log.json"))
    if not dirs:
        raise SystemExit(f"No results/<model>__<labels>/train_log.json under {src}")
    for mdir in dirs:
        log = json.load(open(mdir / "train_log.json"))
        env = json.load(open(mdir / "env.json")) if (mdir / "env.json").exists() else {}
        model, labels = log["model"], log["labels"]
        for corpus, n in EXPECTED.items():
            f = mdir / f"{corpus}.jsonl"
            if not f.exists():
                continue
            rows = {}
            for line in open(f, encoding="utf-8"):
                r = json.loads(line); rows[r["id"]] = r
            if len(rows) < n:
                print(f"skip {model} {labels} {corpus}: {len(rows)} of {n} documents"); continue
            os.environ["EVAL_DATASET"] = corpus
            import importlib, corpora
            importlib.reload(corpora)
            out = corpora.preds_dir() / f"ft_scec__{model.replace('/', '__')}__{labels}.jsonl"
            meta = {"_meta": {"name": f"{model} fine-tuned on S-CECILIA ({labels})",
                              "taxonomy": "scec" if labels == "native" else "native",
                              "family": "WP17 fine-tuned on S-CECILIA", "device": env.get("device"),
                              "best_epoch": log.get("best_epoch"), "epochs": log.get("epochs"),
                              "train_docs": log.get("train_docs"), "val_docs": log.get("val_docs"),
                              "train_seconds": log.get("train_seconds"), "lr": log.get("lr"),
                              "source": "kits/ft_scec"}}
            with open(out, "w", encoding="utf-8") as g:
                g.write(json.dumps(meta) + "\n")
                for r in rows.values():
                    g.write(json.dumps({"id": r["id"], "spans": r["spans"]}) + "\n")
            print(f"wrote {out.relative_to(HERE)} ({len(rows)} documents)")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1])
