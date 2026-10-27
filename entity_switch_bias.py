"""National-origin bias on the entity-switched datasets of Agarwal et al.
(2020), "Entity-Switched Datasets: An Approach to Auditing the In-Domain
Robustness of Named Entity Recognition Models".

The authors swap every entity of the CoNLL-2003 test set (eng.testb) for an
entity of the same type from another country, so the context stays fixed
and only the entity changes. Their released lists (Indian, Vietnamese;
third_party/entity_switched_src, commit 14a0434, data only) are applied in
the order their switch_entities.py applies them: one list entry per entity,
an entity being a run of tokens with the same type. The CoNLL-2003 test set
is tner/conll2003 dataset/test.json (data files only; 3,453 sentences, the
5,628 entities the lists expect).

Each sentence (tokens joined by spaces) goes through the release pipeline
twice, as in kits/mehrabi_full/run.py: general model "simple" (before the
whole-word fix) and "first" (after), the other detectors shared.
Per gold entity (PER, LOC, ORG; MISC kept for the general model only):
  redacted  fully covered by redacted (non-preserved) entities
  typed     one entity of the matching SIESTA type covers it
            (PER->PER, LOC->LOC, ORG->ORG)
and, for the general model alone, strict span-and-type P/R/F1 as in the
paper. CIs: 95% bootstrap over sentences (2,000 resamples).

With --variant siesta-fix<...> (fixes 5-7, as in run_siesta.py), the
pipeline runs once, "first" only, with that fix's constants set, into
logs/entity_switch_bias_<variant>.jsonl; the report then compares it with
the "first" run of logs/entity_switch_bias.jsonl (the working tree with the
whole-word fix, before fixes 5-7) and goes to ENTITY_SWITCH_BIAS_<variant>.md.

Usage: [SIESTA_DEVICE=0] entity_switch_bias.py [--report] [--variant siesta-fix...]
"""
import json, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "third_party"
REPO = HERE / "siesta_aad6c5b_fixC" / "Anonimizacion"
BASE_OUT = HERE / "logs" / "entity_switch_bias.jsonl"
FIX = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else None
OUT = HERE / "logs" / f"entity_switch_bias_{FIX}.jsonl" if FIX else BASE_OUT
SETS = {"original": None, "indian": "entities-indian", "vietnamese": "entities-viet"}
VARIANTS = ("first",) if FIX else ("simple", "first")
TYPE_MAP = {"PER": "PER", "LOC": "LOC", "ORG": "ORG"}


def conll():
    lab = json.load(open(SRC / "conll2003_src" / "label.json"))
    inv = {v: k for k, v in lab.items()}
    sents = []
    for line in open(SRC / "conll2003_src" / "test.json", encoding="utf-8"):
        r = json.loads(line)
        sents.append([(tok, inv[t][2:] if inv[t] != "O" else "O") for tok, t in zip(r["tokens"], r["tags"])])
    return sents


def switched(sents, list_name):
    """Replace each entity run, in order, with the next list entry (same
    counting as switch_entities.py: a new entity starts where the type
    changes)."""
    if list_name is None:
        return sents
    repl = [l.strip() for l in open(SRC / "entity_switched_src" / "repo" / "entitites" / list_name, encoding="utf-8")]
    k, out = 0, []
    for s in sents:
        new, i = [], 0
        while i < len(s):
            tok, ty = s[i]
            if ty == "O":
                new.append((tok, ty)); i += 1; continue
            j = i
            while j < len(s) and s[j][1] == ty:
                j += 1
            new += [(w, ty) for w in repl[k].split()]
            k += 1; i = j
        out.append(new)
    assert k == len(repl), (k, len(repl))
    return out


def spans(sent):
    """Text and gold entity spans [(start, end, type)] for a token list."""
    text, pos, ents, cur = "", 0, [], None
    for tok, ty in sent:
        if text:
            text += " "; pos += 1
        s, e = pos, pos + len(tok)
        text += tok; pos = e
        if ty != "O" and cur and cur[2] == ty:
            cur[1] = e
        else:
            if cur: ents.append(tuple(cur))
            cur = [s, e, ty] if ty != "O" else None
    if cur: ents.append(tuple(cur))
    return text, ents


def build():
    sys.path.insert(0, str(REPO)); os.chdir(REPO)
    import torch, base_anonymizer, custom_entities
    from transformers import pipeline
    from en_anonymizer import EnglishAnonymizer
    if FIX:  # same constants as run_siesta.build
        base_anonymizer.FAIL_CLOSED_NAMES = "6" in FIX.split("-")[1]
        base_anonymizer.CYBER_MIN_SCORE = float(FIX.split("-t")[1]) if "-t" in FIX else None
    a = EnglishAnonymizer()
    old = a.pipelines["general"]
    pipes = {v: pipeline("ner", model=old.model, tokenizer=old.tokenizer, aggregation_strategy=v,
                         stride=base_anonymizer.NER_STRIDE, device=-1) for v in VARIANTS}
    if int(os.environ.get("SIESTA_DEVICE", "-1")) >= 0:
        for p in list(pipes.values()) + ([a.cisco_pipe] if a.cisco_pipe else []):
            p.model.to("cuda"); p.device = torch.device("cuda")
        if custom_entities._load() is not None:
            custom_entities._model = custom_entities._model.to("cuda")
    cache = {}

    def cached(key, fn):
        def wrapper(*args):
            k = (key, args[0])
            if k not in cache:
                cache[k] = fn(*args)
            return [dict(e) for e in cache[k]]
        return wrapper
    a.detect_cyber_entities = cached("cyber", a.detect_cyber_entities)
    a.detect_dates = cached("dates", a.detect_dates)
    a.detect_regex = cached("regex", a.detect_regex)
    custom_entities.detect = cached("gliner", custom_entities.detect)
    return a, pipes, cache


def run():
    base = conll()
    a, pipes, cache = build()
    t0 = time.time()
    with open(OUT.with_suffix(".part"), "w", encoding="utf-8") as f:
        for name, lst in SETS.items():
            for i, sent in enumerate(switched(base, lst)):
                text, gold = spans(sent)
                rec = {"set": name, "i": i, "gold": gold}
                cache.clear()
                for v in VARIANTS:
                    a.pipelines["general"] = pipes[v]
                    _, ents, _ = a.anonymize(text, 0) if text.strip() else (None, [], None)
                    gen = [(x["start"], x["end"], x["entity_group"]) for x in pipes[v](text)] if text.strip() else []
                    rec[v] = {"final": [(x["start"], x["end"], x.get("readable_type"), bool(x.get("preserved"))) for x in ents],
                              "general": gen}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 500 == 0:
                    print(name, i, f"{time.time() - t0:.0f}s", flush=True)
    OUT.with_suffix(".part").rename(OUT)


def entity_outcomes(rec, v):
    """Per gold entity: (type, redacted, typed)."""
    out = []
    for s, e, ty in rec["gold"]:
        red = set()
        for a_, b, _, pres in rec[v]["final"]:
            if not pres:
                red.update(range(max(a_, s), min(b, e)))
        n = sum(1 for c in range(s, e))
        typed = any(t == TYPE_MAP.get(ty) and a_ <= s and b >= e for a_, b, t, _ in rec[v]["final"])
        out.append((ty, len(red) >= n, typed))
    return out


def general_prf(rec, v):
    """Strict span+type counts for the general model alone (PER/LOC/ORG/MISC)."""
    gold = {(s, e, t) for s, e, t in rec["gold"]}
    pred = {(s, e, t) for s, e, t in rec[v]["general"]}
    return len(gold & pred), len(pred), len(gold)


def clusters(recs, texts, st, ty, metric):
    """Occurrences of each distinct entity string in one set, as arrays of
    per-cluster sums: (sum simple, sum first, occurrences). metric "leak" =
    not fully redacted; "typed" = given the right type."""
    import numpy as np
    d = {}
    for r in recs:
        if r["set"] != st:
            continue
        t = texts[st][r["i"]]
        for g, a, b in zip(r["gold"], entity_outcomes(r, "simple"), entity_outcomes(r, "first")):
            if g[2] != ty:
                continue
            val = (lambda x: not x[1]) if metric == "leak" else (lambda x: x[2])
            d.setdefault(t[g[0]:g[1]], []).append((val(a), val(b)))
    c = list(d.values())
    return (np.array([sum(x[0] for x in v) for v in c], float), np.array([sum(x[1] for x in v) for v in c], float),
            np.array([len(v) for v in c], float))


def report(recs):
    import numpy as np
    rng = np.random.default_rng(0)
    base = conll()
    texts = {st: [spans(s)[0] for s in switched(base, lst)] for st, lst in SETS.items()}
    what = (f"`Before fix` = the working tree with the whole-word fix (before fixes 5-7), `After fix` = {FIX}."
            if FIX else "`simple` = SIESTA before the whole-word fix, `first` = after.")
    L = ["# National-origin bias: entity-switched CoNLL-2003 test set (Agarwal et al., 2020)", "",
         "Same 3,453 news sentences; every entity swapped for an Indian or a Vietnamese one of the same type "
         f"(the authors' released lists). {what}", "",
         "95% CIs resample distinct entity strings (2,000 resamples), keeping all occurrences of a string together: "
         "the lists repeat entities (884 distinct strings in 5,620 Indian entries), and one string's occurrences "
         "share most of their outcome. The original set is CoNLL-2003's own mixed, mostly Western entities, "
         "not one nationality.", ""]

    def boot(cl, n=2000):
        idx = rng.integers(0, len(cl[2]), size=(n, len(cl[2])))
        s, f, k = cl[0][idx].sum(1), cl[1][idx].sum(1), cl[2][idx].sum(1)
        return s / k, f / k

    def ci(v):
        v = np.sort(v); return f"({100 * v[int(.025 * len(v))]:.1f}, {100 * v[int(.975 * len(v)) - 1]:.1f})"

    def cip(v):
        v = np.sort(v); return f"({100 * v[int(.025 * len(v))]:+.1f}, {100 * v[int(.975 * len(v)) - 1]:+.1f})"

    for metric, title in (("leak", "Entities left partly unredacted (leak), %"), ("typed", "Entities given the right type, %")):
        L += [f"## {title}", "",
              "| Entity | Set | Distinct / occurrences | Before fix (CI) | After fix (CI) | Change from fix, points (CI) | Gap vs original after fix, points (CI) |",
              "|---|---|---|---|---|---|---|"]
        for ty in ("PER", "LOC", "ORG"):
            cl0 = clusters(recs, texts, "original", ty, metric); _, f0 = boot(cl0)
            for st in SETS:
                cl = clusters(recs, texts, st, ty, metric)
                s, f = boot(cl)
                ms, mf = cl[0].sum() / cl[2].sum(), cl[1].sum() / cl[2].sum()
                gap = "–" if st == "original" else f"{100 * (mf - cl0[1].sum() / cl0[2].sum()):+.1f} {cip(f - f0)}"
                L.append(f"| {ty} | {st} | {len(cl[2])} / {int(cl[2].sum())} | {100 * ms:.1f} {ci(s)} | {100 * mf:.1f} {ci(f)} | "
                         f"{100 * (mf - ms):+.1f} {cip(f - s)} | {gap} |")
        L.append("")
    if FIX:  # the general model is the same before and after fixes 5-7
        (HERE / f"ENTITY_SWITCH_BIAS_{FIX}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))
        return
    L += ["## General model alone, strict span-and-type F1 (as in the paper)", "", "| Set | " + " | ".join(VARIANTS) + " |",
          "|---|" + "---|" * len(VARIANTS)]
    for st in SETS:
        rs = [r for r in recs if r["set"] == st]
        cells = []
        for v in VARIANTS:
            tp = sum(general_prf(r, v)[0] for r in rs); np_ = sum(general_prf(r, v)[1] for r in rs); ng = sum(general_prf(r, v)[2] for r in rs)
            p, rc = tp / max(np_, 1), tp / max(ng, 1)
            cells.append(f"{2 * p * rc / max(p + rc, 1e-9):.3f} (P {p:.3f}, R {rc:.3f})")
        L.append(f"| {st} | " + " | ".join(cells) + " |")
    (HERE / "ENTITY_SWITCH_BIAS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def load():
    recs = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    if FIX:  # baseline "first" as "simple", the fix as "first"
        base = {(r["set"], r["i"]): r for r in map(json.loads, BASE_OUT.read_text(encoding="utf-8").splitlines())}
        recs = [dict(r, simple=base[(r["set"], r["i"])]["first"]) for r in recs]
    return recs


if __name__ == "__main__":
    if not ("--report" in sys.argv and OUT.exists()):
        run()
    report(load())
