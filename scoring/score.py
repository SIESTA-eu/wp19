"""Score every preds/<system>.jsonl against S-CECILIA gold.

Metrics
  exact    type-agnostic span P/R/F1, identical boundaries only ("whole"): a partial
           detection is an error (RP1 ethics review recommendation).
  detect   type-agnostic span P/R/F1, 1:1 greedy match at IoU >= 0.5 ("partial").
           "Was the sensitive thing found?" -- the anonymisation question.
  leak     share of gold-entity characters (non-space) no prediction covers.
  conll    span P/R/F1 at IoU >= 0.5 with matching CoNLL-4 type, using
           WP17's CECILIA -> CoNLL collapse (PER / ORG / LOC / MISC).
  strict   31-label P/R/F1, same label and IoU >= 0.8. Only for
           systems that emit the S-CECILIA taxonomy.

Boundaries are compared after trimming whitespace and trailing
punctuation from both sides: the gold has "Bandon." and "45.1.2.3:443."

Usage: score.py [corpus] [subset]
  corpus  scec (default) | cecilia | anonlfi_cases | tab | cyner
  subset  test (default) | train | all   (anonlfi_cases: case1 | case2 | all)

On S-CECILIA test, detect F1 also gets a 95% interval from 2,000 bootstrap
resamples of the held-out seeds (whole seeds, since reports within a seed
are alike).
"""
import collections, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
TRIM = " \t\n.,;:!?'\"()[]{}"

SCEC_TO_CONLL = {
    "Person_Name": "PER", "Organization": "ORG", "Organisation": "ORG",
    "Geographic_Location": "LOC", "Country": "LOC", "Facility": "LOC",
}
CONLL = {"PER", "ORG", "LOC", "MISC"}

# S-CECILIA types by what they are, for the per-group breakdown. Threat
# intelligence is public and identifies no one: it is scored for detection
# but left out of the privacy leak ("leak_private").
GROUPS = {
    "technical": {"IP_Address", "URL_Domain", "MAC_Address", "System_Identifiers", "File_Resources",
                  "Cryptographic_Elements", "Software_Information", "Operating_System"},
    "threat_intel": {"Vulnerability_Identifiers", "Malware_Name", "Threat_Actor"},
}
def group_of(label):
    for g, types in GROUPS.items():
        if label in types: return g
    return "personal"
# Native labels of each system family -> CoNLL-4 (anything else -> MISC)
NATIVE_TO_CONLL = {
    "PER": "PER", "PERSON": "PER", "ORG": "ORG", "ORGANIZATION": "ORG",
    "LOC": "LOC", "GPE": "LOC", "FAC": "LOC", "LOCATION": "LOC",
}


CECILIA_TO_CONLL = {  # WP17 ner-finetune.py
    "Person": "PER", "Facilities": "LOC", "GPE_Location": "LOC", "Location": "LOC", "Address": "LOC",
    "Organization": "ORG", "Group": "ORG"}


def to_conll(label, taxonomy):
    if taxonomy == "scec":
        return SCEC_TO_CONLL.get(label, "MISC")
    if taxonomy == "cecilia":
        return CECILIA_TO_CONLL.get(label, "MISC")
    return NATIVE_TO_CONLL.get(label.upper(), "MISC")


def trim(text, s, e):
    e = min(e, len(text))  # GLiNER2 once returned end=3 for the 2-char text "OS"
    while s < e and text[s] in TRIM: s += 1
    while e > s and text[e - 1] in TRIM: e -= 1
    return s, e


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]) or 1)


def match(pred, gold_, thr, same=lambda p, g: True):
    """Greedy 1:1 by descending IoU. Returns (tp, matched gold indices)."""
    cand = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold_)
                   if same(p, g)), reverse=True)
    up, ug = set(), set()
    for v, i, j in cand:
        if v < thr: break
        if i in up or j in ug: continue
        up.add(i); ug.add(j)
    return len(up), ug


def prf(tp, np_, ng):
    p = tp / np_ if np_ else 0.0; r = tp / ng if ng else 0.0
    return dict(p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0.0)


def load_preds(path):
    meta, rows = None, {}
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        if "_meta" in o: meta = o["_meta"]; continue
        rows[o["id"]] = o["spans"]
    return meta, rows


def score(system_path, ids, docs, gold_tax="scec", groups=None):
    meta, P = load_preds(system_path)
    tax = meta.get("taxonomy", "native")
    per_doc = {}
    agg = collections.Counter()
    per_type = collections.defaultdict(collections.Counter)
    ch_tot = ch_hit = ch_pred = ch_pred_in = 0
    grp = collections.Counter()  # per-group span hits and character coverage
    missing = 0
    for gid in ids:
        d = docs[gid]; text = d["text"]
        if gid not in P: missing += 1
        g = [(*trim(text, s, e), l) for s, e, l in d["spans"]]
        g = [x for x in g if x[1] > x[0]]
        p = [(*trim(text, s, e), l) for s, e, l in P.get(gid, [])]
        p = [x for x in p if x[1] > x[0]]
        # detect
        tp, ug = match(p, g, 0.5)
        agg["d_tp"] += tp; agg["np"] += len(p); agg["ng"] += len(g)
        # exact: identical boundaries (after trimming). A partial detection
        # counts as an error, as the RP1 ethics review asks.
        agg["x_tp"] += match(p, g, 1.0)[0]
        per_doc[gid] = (tp, len(p), len(g))
        for j, x in enumerate(g):
            per_type[x[2]]["n"] += 1; per_type[x[2]]["hit"] += j in ug
            if gold_tax == "scec":
                k = group_of(x[2]); grp[k + "_n"] += 1; grp[k + "_hit"] += j in ug
        # conll
        pc = [(s, e, to_conll(l, tax)) for s, e, l in p]
        gc = [(s, e, to_conll(l, gold_tax)) for s, e, l in g]
        tp, _ = match(pc, gc, 0.5, lambda a, b: a[2] == b[2])
        agg["c_tp"] += tp
        # strict 31-label
        if tax == "scec" and gold_tax == "scec":
            norm = lambda l: "Organization" if l == "Organisation" else l
            tp, _ = match(p, g, 0.8, lambda a, b: norm(a[2]) == norm(b[2]))
            agg["s_tp"] += tp
        # characters
        gm = bytearray(len(text)); pm = bytearray(len(text)); gg = [None] * len(text)
        for s, e, l in g:
            gm[s:e] = b"\1" * (e - s)
            if gold_tax == "scec": gg[s:e] = [group_of(l)] * (e - s)
        for s, e, _ in p: pm[s:e] = b"\1" * (e - s)
        for i, ch in enumerate(text):
            if ch.isspace(): continue
            if gm[i]:
                ch_tot += 1; ch_hit += pm[i]
                if gg[i]: grp[gg[i] + "_ch"] += 1; grp[gg[i] + "_chhit"] += pm[i]
            if pm[i]:
                ch_pred += 1; ch_pred_in += gm[i]
    out = dict(system=meta["name"], taxonomy=tax, docs=len(ids), missing=missing,
               detect=prf(agg["d_tp"], agg["np"], agg["ng"]),
               exact=prf(agg["x_tp"], agg["np"], agg["ng"]),
               conll=prf(agg["c_tp"], agg["np"], agg["ng"]),
               leak=1 - ch_hit / ch_tot, char_precision=ch_pred_in / ch_pred if ch_pred else 0.0,
               n_pred=agg["np"], n_gold=agg["ng"],
               per_type={k: v["hit"] / v["n"] for k, v in per_type.items()},
               per_type_n={k: v["n"] for k, v in per_type.items()})
    if tax == "scec" and gold_tax == "scec":
        out["strict"] = prf(agg["s_tp"], agg["np"], agg["ng"])
    if gold_tax == "scec":
        out["groups"] = {k: dict(n=grp[k + "_n"], recall=grp[k + "_hit"] / grp[k + "_n"] if grp[k + "_n"] else 0.0,
                                 leak=1 - grp[k + "_chhit"] / grp[k + "_ch"] if grp[k + "_ch"] else 0.0)
                         for k in ("personal", "technical", "threat_intel")}
        priv = grp["personal_ch"] + grp["technical_ch"]
        out["leak_private"] = 1 - (grp["personal_chhit"] + grp["technical_chhit"]) / priv if priv else 0.0
    if groups:
        out["detect_f1_ci95"] = bootstrap(per_doc, groups)
    out.update({k: v for k, v in meta.items() if k not in ("name", "taxonomy")})
    return out


def bootstrap(per_doc, groups, n=2000):
    by = collections.defaultdict(list)
    for gid, v in per_doc.items(): by[groups[gid]].append(v)
    keys = sorted(by); rng = random.Random(0); f1s = []
    for _ in range(n):
        tp = np_ = ng = 0
        for k in (rng.choice(keys) for _ in keys):
            for a, b, c in by[k]: tp += a; np_ += b; ng += c
        f1s.append(prf(tp, np_, ng)["f1"])
    f1s.sort()
    return [f1s[int(0.025 * n)], f1s[int(0.975 * n)]]


def main(corpus="scec", subset="test"):
    os.environ["EVAL_DATASET"] = corpus
    import corpora
    docs = {d["id"]: d for d in corpora.load()}
    if corpus == "anonlfi_cases":
        ids = [i for i in docs if subset == "all" or i.startswith(subset)]
    elif corpus in ("tab", "cyner"):
        ids = list(docs)
    else:
        split = corpora.split()
        ids = split[subset] if subset in split else list(docs)
    groups = corpora.split()["gold_seed"] if corpus == "scec" and subset == "test" else None
    valid = lambda r: subset in r.get("valid_subsets", [subset]) or corpus != "scec"
    res = [score(p, ids, docs, corpora.gold_taxonomy(), groups) for p in sorted(corpora.preds_dir().glob("*.jsonl"))]
    res = [r for r in res if valid(r)]
    res.sort(key=lambda r: -r["detect"]["f1"])
    json.dump(res, open(ROOT / f"results_{corpus}_{subset}.json", "w"), indent=1)
    print(f"{corpus} / {subset}: {len(ids)} docs, {res[0]['n_gold'] if res else 0} gold spans")
    print(f"{'system':44}{'exactF1':>8}{'det P':>7}{'det R':>7}{'det F1':>7}{'  95% CI':>14}{'leak':>7}{'CoNLL F1':>9}{'strict F1':>10}{'exact P':>9}{'exact R':>8}")
    # exact P / R appended at the end of the line: earlier logs read the columns
    # before them by position.
    for r in res:
        st = f"{r['strict']['f1']:10.3f}" if "strict" in r else f"{'-':>10}"
        ci = "  [{:.2f},{:.2f}]".format(*r["detect_f1_ci95"]) if "detect_f1_ci95" in r else " " * 14
        print(f"{r['system'][:43]:44}{r['exact']['f1']:8.3f}{r['detect']['p']:7.3f}{r['detect']['r']:7.3f}{r['detect']['f1']:7.3f}{ci}"
              f"{r['leak']:7.3f}{r['conll']['f1']:9.3f}{st}{r['exact']['p']:9.3f}{r['exact']['r']:8.3f}" + (f"  (missing {r['missing']})" if r["missing"] else ""))
    if res and "groups" in res[0]:
        n = res[0]["groups"]
        print(f"\nRecall / leak by group  (personal n={n['personal']['n']}, technical n={n['technical']['n']}, "
              f"threat intel n={n['threat_intel']['n']}; private leak excludes threat intel)")
        print(f"{'system':44}{'pers R':>8}{'pers leak':>10}{'tech R':>8}{'tech leak':>10}{'TI R':>7}{'priv leak':>10}")
        for r in res:
            G = r["groups"]
            print(f"{r['system'][:43]:44}{G['personal']['recall']:8.3f}{G['personal']['leak']:10.3f}"
                  f"{G['technical']['recall']:8.3f}{G['technical']['leak']:10.3f}{G['threat_intel']['recall']:7.3f}{r['leak_private']:10.3f}")


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
