"""Phase 2 of the GLiNER2 experiment: which fixes help SIESTA + GLiNER2?

Replays SIESTA's own merge (BaseAnonymizer.clean_entities) on cached raw
detections (gliner_cache.py), so each configuration costs seconds. Every
choice is made on the S-CECILIA training seeds; the held-out seeds are
scored once, for the final configurations only.

Fixes (IMPROVEMENTS.md, section A)
  G1 snap AGE to "age N" / "N-year-old" / "N years old" / "aged N"
  G2 drop category words for value types (no digit + a concept noun)
  G3 per-type thresholds, coordinate ascent on training-seed F1
  G4 drop vague times and generic nouns as SOFTWARE
  G5 windowed input (cache mode "windowed")
  G6 gliner2-large-v1

  gliner_exp.py            the comparison above (results_gliner_exp.json)
  gliner_exp.py deployed   the thresholds deployed in custom_entities.INCIDENT_REPORT_PACK
                           (results_gliner_base_tuned.json)
"""
import collections, copy, json, pickle, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
import gold, score  # noqa: E402
import run_siesta  # noqa: E402
from base_anonymizer import BaseAnonymizer  # noqa: E402  (SIESTA repo, via run_siesta)

TYPES = {c["name"]: c["name"].upper().replace(" ", "_") for c in run_siesta.CUSTOM_TYPES}
VALUE_TYPES = {"username", "password", "identification number", "bank or card number", "phone number"}
CONCEPT = re.compile(r"\b(user ?names?|passwords?|pins?|credentials?|logins?|social security|driver'?s licen[cs]es?|"
                     r"passports?|ids?|identification|numbers?|cards?|credit|expiry|cvvs?|phones?|telephones?|"
                     r"accounts?|addresses|emails?)\b", re.I)
VAGUE_TIME = re.compile(r"^(the )?(morning|afternoon|evening|night|midnight|noon|dawn|dusk|sunset|sunrise|"
                        r"today|tonight|yesterday|daily|weekly|hours?|minutes?|days?|weeks?|months?|years?)$", re.I)
GENERIC_SW = re.compile(r"^(the )?(malware|ransomware|spyware|software|virus|trojan|worm|botnet|antivirus|"
                        r"security software|antivirus software|app|apps|application|firmware|code)$", re.I)


class Merger(BaseAnonymizer):
    """clean_entities without loading any model."""
    def __init__(self):
        pass


M = Merger()
docs = {d["id"]: d for d in gold.load()}
split = json.load(open(ROOT / "corpus_loaders" / "split.json"))
raw = pickle.load(open(ROOT / "cache" / "siesta_raw_scec.pkl", "rb"))


def snap_age(text, s, e):
    """G1: widen a bare number to the age expression around it."""
    left, right = text[max(0, s - 8):s], text[e:e + 14]
    m = re.search(r"(?:\bage[ds]?\s*:?\s*|\baged\s+)$", left, re.I)
    if m: s -= len(m.group(0))
    m = re.match(r"(?:-year-old|\s+years?\s+old|\s*-?\s*years?\b)", right, re.I)
    if m: e += len(m.group(0))
    return s, e


def custom_spans(text, found, thr, fixes):
    out = []
    for s, e, name, conf in found:
        if name not in TYPES or conf < thr.get(name, 0.5): continue
        x = text[s:e].strip()
        if not x: continue
        if "G2" in fixes and name in VALUE_TYPES and not re.search(r"\d", x) and CONCEPT.search(x): continue
        if "G4" in fixes and name == "time" and VAGUE_TIME.match(x): continue
        if "G4" in fixes and name == "software" and GENERIC_SW.match(x): continue
        if "G1" in fixes and name == "age": s, e = snap_age(text, s, e)
        out.append({"start": s, "end": e, "text": text[s:e], "entity_group": TYPES[name], "score": conf,
                    "_source": "custom"})
    return out


def predict(gid, gl, thr, fixes):
    t = docs[gid]["text"]; r = raw[gid]
    ner = copy.deepcopy(r["ner"]); reg = copy.deepcopy(r["reg"]) + custom_spans(t, gl.get(gid, []), thr, fixes)
    return [(e["start"], e["end"], e["entity_group"]) for e in M.clean_entities(t, ner, reg)]


def evaluate(ids, gl, thr, fixes):
    tp = np_ = ng = 0; ch = hit = 0
    for gid in ids:
        t = docs[gid]["text"]
        p = [(*score.trim(t, s, e), l) for s, e, l in predict(gid, gl, thr, fixes)]
        g = [(*score.trim(t, s, e), l) for s, e, l in docs[gid]["spans"]]
        k, _ = score.match(p, g, 0.5); tp += k; np_ += len(p); ng += len(g)
        pm = bytearray(len(t))
        for s, e, _ in p: pm[s:e] = b"\1" * (e - s)
        for s, e, l in g:
            if score.group_of(l) == "threat_intel": continue
            for i in range(s, e):
                if not t[i].isspace(): ch += 1; hit += pm[i]
    r = score.prf(tp, np_, ng); r["leak"] = 1 - hit / ch
    return r


def tune(ids, gl, fixes, grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """G3: per-type thresholds by coordinate ascent on detection F1 (two passes)."""
    thr = {n: 0.5 for n in TYPES}; best = evaluate(ids, gl, thr, fixes)["f1"]
    for _ in range(2):
        for n in TYPES:
            for v in grid:
                trial = {**thr, n: v}; f = evaluate(ids, gl, trial, fixes)["f1"]
                if f > best + 1e-4: best, thr = f, trial
    return thr


def main():
    base = {n: 0.5 for n in TYPES}
    caches = {m: pickle.load(open(ROOT / "cache" / f"gliner_{m}.pkl", "rb"))
              for m in ["gliner2-base-v1_full", "gliner2-base-v1_windowed", "gliner2-large-v1_windowed"]
              if (ROOT / "cache" / f"gliner_{m}.pkl").exists()}
    tr, te = split["train"], split["test"]
    rows = []
    def row(name, cache, thr, fixes, test=False):
        a = evaluate(tr, caches[cache], thr, fixes)
        b = evaluate(te, caches[cache], thr, fixes) if test else None
        rows.append(dict(config=name, cache=cache, fixes=sorted(fixes), thresholds=thr, train=a, test=b))
        tb = f"   test {b['p']:.3f}/{b['r']:.3f}/{b['f1']:.3f}/{b['leak']:.3f}" if b else ""
        print(f"{name:44} train {a['p']:.3f}/{a['r']:.3f}/{a['f1']:.3f}/{a['leak']:.3f}{tb}", flush=True)

    print("P/R/F1/private leak\n")
    c0 = "gliner2-base-v1_full"
    row("as deployed (base, whole text, thr 0.5)", c0, base, set(), test=True)
    for f in ["G1", "G2", "G4"]:
        row(f"  + {f} only", c0, base, {f})
    row("  + G1 G2 G4", c0, base, {"G1", "G2", "G4"})
    for c in caches:
        if c == c0: continue
        row(f"  + G1 G2 G4, {c}", c, base, {"G1", "G2", "G4"})
    best_cache = max(caches, key=lambda c: evaluate(tr, caches[c], base, {"G1", "G2", "G4"})["f1"])
    thr = tune(tr, caches[best_cache], {"G1", "G2", "G4"})
    print("tuned thresholds:", thr)
    row(f"FINAL: G1 G2 G4 + G3 thresholds, {best_cache}", best_cache, thr, {"G1", "G2", "G4"}, test=True)
    json.dump(rows, open(ROOT / "rubiner" / "thresholds" / "results_gliner_exp.json", "w"), indent=1)


def deployed():
    """The entity-pack thresholds in custom_entities.INCIDENT_REPORT_PACK: G3 on gliner2-base-v1,
    windowed, with G1 G2 G4, tuned on the training seeds; held-out seeds scored once."""
    gl = pickle.load(open(ROOT / "cache" / "gliner_gliner2-base-v1_windowed.pkl", "rb"))
    fixes = {"G1", "G2", "G4"}
    thr = tune(split["train"], gl, fixes)
    out = dict(model="gliner2-base-v1", mode="windowed", fixes=sorted(fixes), thresholds=thr,
               train=evaluate(split["train"], gl, thr, fixes), test=evaluate(split["test"], gl, thr, fixes))
    json.dump(out, open(ROOT / "rubiner" / "thresholds" / "results_gliner_base_tuned.json", "w"), indent=1)
    print(json.dumps(thr))


if __name__ == "__main__":
    deployed() if sys.argv[1:] == ["deployed"] else main()
