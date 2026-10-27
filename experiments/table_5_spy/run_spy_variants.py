"""SPY through RubiNER and the same pipeline without fine-tuning, on 8,688
documents (spy_corpus.py: document generator and scoring).

  run      writes logs/spy_variants.jsonl: per document, gold and the entities of each
           variant as (start, end, readable_type, preserved)
  --report prints per-type redaction rates (glued values excluded) and
           type-agnostic detection P/R/F1, and writes logs/spy_variants.md
           (Table 5: the RubiNER column)

Run from the repository root:
  SIESTA_DEVICE=0 .venv/bin/python experiments/table_5_spy/run_spy_variants.py
"""
import collections, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
OUT = ROOT / "logs" / "spy_variants.jsonl"
VARIANTS = ("fixC", "rubiner")


def run():
    import run_siesta, custom_entities  # run_siesta puts SIESTA_REPO on the path
    import run_rubiner as ft
    import spy_corpus as sb
    from base_anonymizer import windowed_ner_pipeline
    import torch
    custom_entities._load()
    if run_siesta.DEV >= 0 and custom_entities._model is not None:
        custom_entities._model = custom_entities._model.to("cuda")
    fix = run_siesta.build("aad-fixC")
    rub = run_siesta.build("aad-fixC")
    rub.pipelines.pop("general")
    pipe = windowed_ner_pipeline(str(ft.checkpoint("siesta")))
    if run_siesta.DEV >= 0:
        pipe.model.to("cuda"); pipe.device = torch.device("cuda")
    base, extra = rub.detect_ner, ft.ft_detector(pipe)
    rub.detect_ner = lambda text: base(text) + extra(text)
    t0 = time.time()
    with open(OUT.with_suffix(".part"), "w", encoding="utf-8") as f:
        for k, (dom, i, text, gold) in enumerate(sb.documents()):
            rec = {"domain": dom, "i": i, "gold": gold}
            for v, a in (("fixC", fix), ("rubiner", rub)):
                _, ents, _ = a.anonymize(text, 0)
                rec[v] = [(x["start"], x["end"], x.get("readable_type"), bool(x.get("preserved"))) for x in ents]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 500 == 0:
                print(dom, i, f"{time.time() - t0:.0f}s", flush=True)
    OUT.with_suffix(".part").rename(OUT)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


def report():
    os.chdir(ROOT)
    import spy_corpus as sb
    recs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    sb.drop_glued(recs)
    names = {"fixC": "Without fine-tuning", "rubiner": "RubiNER"}
    L = ["# SPY: RubiNER", "",
         "Share of SPY values fully redacted, by domain and type; values glued to a neighbouring word excluded.", "",
         "| Domain | SPY type | n | " + " | ".join(names[v] for v in VARIANTS) + " |", "|---|---|---|---|---|"]
    for dom in ("medical", "legal"):
        rows = [r for r in recs if r["domain"] == dom]
        by = {v: collections.defaultdict(list) for v in VARIANTS}
        for r in rows:
            for v in VARIANTS:
                for ty, red, typed, s, e in sb.clean(r, v):
                    by[v][ty].append(red)
        for ty in sorted(by["rubiner"], key=lambda t: -len(by["rubiner"][t])):
            n = len(by["rubiner"][ty])
            L.append(f"| {dom} | {ty} | {n} | " + " | ".join(f"{100 * sum(by[v][ty]) / n:.1f}%" for v in VARIANTS) + " |")
    L += ["", "Type-agnostic detection over all values (IoU >= 0.5); SPY annotates seven types only, so precision understates every system.", "",
          "| Domain | " + " | ".join(f"{names[v]} P / R / F1" for v in VARIANTS) + " |", "|---|---|---|"]
    for dom in ("medical", "legal"):
        rows = [r for r in recs if r["domain"] == dom]
        L.append(f"| {dom} | " + " | ".join("{:.3f} / {:.3f} / {:.3f}".format(*sb.detect_prf(rows, v)) for v in VARIANTS) + " |")
    txt = "\n".join(L) + "\n"
    (ROOT / "logs" / "spy_variants.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
