"""SPY through SIESTA + corrections and RubiNER, on the same 8,688 documents as
spy_benchmark.py (its document generator and scoring are reused unchanged).

  run      writes logs/spy_variants.jsonl: per document, gold and the entities of each
           variant as (start, end, readable_type, preserved), as spy_benchmark.py records them
  --report prints per-type redaction rates (glued values excluded, as in spy_benchmark.py)
           and type-agnostic detection P/R/F1, against the release ('simple' in
           logs/spy_benchmark.jsonl, i.e. aad6c5b), and writes
           analysis_siesta_fixes/spy_variants.md

Run from WP17/eval:
  SIESTA_DEVICE=0 SIESTA_REPO=$PWD/siesta_aad6c5b_fixC/Anonimizacion \
    .venv/bin/python analysis_siesta_fixes/run_spy_variants.py
"""
import collections, json, os, sys, time
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL)); sys.path.insert(0, str(EVAL / "analysis_siesta_fixes"))
OUT = EVAL / "logs" / "spy_variants.jsonl"
VARIANTS = ("fixC", "rubiner")


def run():
    import run_siesta, custom_entities  # run_siesta puts SIESTA_REPO on the path
    import run_siesta_ft as ft
    import spy_benchmark as sb
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
    os.chdir(EVAL)
    import spy_benchmark as sb
    new = {(r["domain"], r["i"]): r for r in map(json.loads, open(OUT, encoding="utf-8"))}
    recs = []
    for r in map(json.loads, open(EVAL / "logs" / "spy_benchmark.jsonl", encoding="utf-8")):
        n = new[(r["domain"], r["i"])]
        assert n["gold"] == r["gold"], "documents differ from spy_benchmark.py's"
        recs.append({"domain": r["domain"], "i": r["i"], "gold": r["gold"], "release": r["simple"], **{v: n[v] for v in VARIANTS}})
    sb.drop_glued(recs)
    cols = ("release",) + VARIANTS
    L = ["# SPY: release, corrections and RubiNER", "",
         "Share of SPY values fully redacted, by domain and type; values glued to a neighbouring word excluded, as in SPY_BENCHMARK.md.", "",
         "| Domain | SPY type | n | Release | + corrections | RubiNER |", "|---|---|---|---|---|---|"]
    for dom in ("medical", "legal"):
        rows = [r for r in recs if r["domain"] == dom]
        by = {v: collections.defaultdict(list) for v in cols}
        for r in rows:
            for v in cols:
                for ty, red, typed, s, e in sb.clean(r, v):
                    by[v][ty].append(red)
        for ty in sorted(by["release"], key=lambda t: -len(by["release"][t])):
            n = len(by["release"][ty])
            L.append(f"| {dom} | {ty} | {n} | " + " | ".join(f"{100 * sum(by[v][ty]) / n:.1f}%" for v in cols) + " |")
    L += ["", "Type-agnostic detection over all values (IoU >= 0.5); SPY annotates seven types only, so precision understates every system.", "",
          "| Domain | Release P / R / F1 | + corrections P / R / F1 | RubiNER P / R / F1 |", "|---|---|---|---|"]
    for dom in ("medical", "legal"):
        rows = [r for r in recs if r["domain"] == dom]
        L.append(f"| {dom} | " + " | ".join("{:.3f} / {:.3f} / {:.3f}".format(*sb.detect_prf(rows, v)) for v in cols) + " |")
    txt = "\n".join(L) + "\n"
    (EVAL / "analysis_siesta_fixes" / "spy_variants.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
