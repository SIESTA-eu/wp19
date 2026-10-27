"""Phase 1 of the GLiNER2 experiment: cache raw detections for every S-CECILIA report.

  cache/siesta_raw_scec.pkl       SIESTA's built-in detections before merging
                                  (general NER, cyber NER, regexes), from the
                                  siesta-stride build. Acronym expansion is
                                  skipped so offsets are on the original text.
  cache/gliner_<model>_<mode>.pkl GLiNER2 spans with confidence, threshold 0.3,
                                  for the 14 CUSTOM_TYPES, called exactly as
                                  custom_entities.detect() calls it.
      mode "full"      the whole report in one call (as deployed)
      mode "windowed"  250-word windows, 50-word overlap (G5)

Phase 2 (gliner_exp.py) replays SIESTA's clean_entities merge on these.
"""
import pickle, re, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gold  # noqa: E402
import run_siesta  # noqa: E402  (chdirs into the SIESTA repo, puts it on sys.path)

CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
WIN, OVERLAP = 250, 50


def siesta_raw(docs):
    path = CACHE / "siesta_raw_scec.pkl"
    if path.exists(): return
    a = run_siesta.build("siesta-stride"); out = {}; t0 = time.time()
    for i, d in enumerate(docs):
        t = d["text"]
        out[d["id"]] = dict(ner=a.detect_ner(t) + a.detect_cyber_entities(t), reg=a.detect_regex(t))
        if i % 200 == 0: print("siesta raw", i, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(out, open(path, "wb"))


def gliner(docs, model_name, mode):
    path = CACHE / f"gliner_{model_name.split('/')[-1]}_{mode}.pkl"
    if path.exists(): return
    from gliner2 import GLiNER2, Schema
    model = GLiNER2.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    schema = Schema().entities({c["name"]: c.get("description") or c["name"] for c in run_siesta.CUSTOM_TYPES})
    out = {}; t0 = time.time()
    for i, d in enumerate(docs):
        t = d["text"]
        if mode == "full":
            chunks = [(0, len(t))]
        else:
            words = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
            chunks = []
            for k in range(0, max(len(words), 1), WIN - OVERLAP):
                w = words[k:k + WIN]
                if not w: break
                chunks.append((w[0][0], w[-1][1]))
                if k + WIN >= len(words): break
        found = []
        for a, b in chunks:
            with torch.no_grad():
                r = model.extract(t[a:b], schema, threshold=0.3, include_spans=True, include_confidence=True)
            for label, spans in (r.get("entities") or {}).items():
                for s in spans or []:
                    found.append((a + s["start"], min(a + s["end"], len(t)), label, float(s.get("confidence", 0.0))))
        # overlapping windows: keep the higher-confidence copy of a duplicate
        best = {}
        for s, e, l, c in found:
            if (s, e, l) not in best or c > best[(s, e, l)]: best[(s, e, l)] = c
        out[d["id"]] = [(s, e, l, c) for (s, e, l), c in best.items()]
        if i % 200 == 0: print(path.name, i, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(out, open(path, "wb"))
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    docs = gold.load()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "siesta"): siesta_raw(docs)
    if what in ("all", "gliner"):
        gliner(docs, "fastino/gliner2-base-v1", "full")
        gliner(docs, "fastino/gliner2-base-v1", "windowed")
        gliner(docs, "fastino/gliner2-large-v1", "windowed")
