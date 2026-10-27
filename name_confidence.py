"""Is the name problem a confidence problem?

For every name in the two bias tests (mehrabi_bias.py, 2018 sample;
template_name_bias.py), records what each detector said about the name and
how sure it was, then what the release pipeline finally did with it:

  general   dslim/bert-base-NER, aggregated as SIESTA runs it: label and
            score of the entity covering the most of the name. SIESTA drops
            these below 0.65 (en_anonymizer.detect_ner).
  p_per     the same model's per-token probability of a person tag
            (B-PER + I-PER), mean and minimum over the name's sub-word
            tokens: how much "person" there was behind the top label.
  cyber     SecureBERT-NER labels and scores on the name.
  gliner    the incident-report pack's findings on the name, as SIESTA sees
            them (custom_entities.detect: per-type thresholds applied).
  final     the type the pipeline gave the name and whether it leaked.

Answers three questions in NAME_CONFIDENCE.md:
  1. How many leaks or mistyped names did the 0.65 cut-off cause?
  2. When the top label is wrong, how much person probability was there?
  3. When conflict resolution overrode a person tag, how sure was the winner?

Usage: [SIESTA_DEVICE=0] name_confidence.py [--report]
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mehrabi_bias as mb
import template_name_bias as tb

OUT = HERE / "logs" / "name_confidence.jsonl"
CUTOFF = 0.65  # en_anonymizer.detect_ner THRESHOLD


def sentences():
    """(test, group, gender, name, text, start, end) for both tests."""
    for g in ("F", "M"):
        rnd, top, freq = mb.sample(g)
        for t in mb.TEMPLATES:
            for name in sorted(rnd | top):
                text = name if t == 1 else name + {4: " is a person", 8: " is a nurse", 9: " is a doctor"}[t]
                yield dict(test="mehrabi", group=f"T{t}", gender=g, name=name, text=text, s=0, e=len(name),
                           freq=freq[name], random=name in rnd, top=name in top)
    for origin, d in tb.NAMES.items():
        for g in ("F", "M"):
            for i, first in enumerate(d[g]):
                for form, name, temps in (("full", f"{first} {d['S'][i]}", tb.TEMPLATES),
                                          ("first", first, tb.FIRST_ONLY_TEMPLATES)):
                    for t in temps:
                        text = t.replace("{N}", name)
                        s = text.index(name)
                        yield dict(test="template", group=origin, form=form, gender=g, name=name, text=text,
                                   s=s, e=s + len(name))


def overlapping(ents, s, e):
    return [x for x in ents if x["start"] < e and x["end"] > s]


def best_cover(ents, s, e):
    """The entity covering most of [s, e), or None."""
    ov = overlapping(ents, s, e)
    if not ov:
        return None
    return max(ov, key=lambda x: min(x["end"], e) - max(x["start"], s))


def run():
    sys.path.insert(0, str(mb.REPO)); os.chdir(mb.REPO)
    import torch
    import custom_entities
    from en_anonymizer import EnglishAnonymizer
    a = EnglishAnonymizer()
    dev = "cuda" if int(os.environ.get("SIESTA_DEVICE", "-1")) >= 0 else "cpu"
    if dev == "cuda":
        for pipe in list(a.pipelines.values()) + ([a.cisco_pipe] if a.cisco_pipe else []):
            pipe.model.to("cuda"); pipe.device = torch.device("cuda")
        if custom_entities._load() is not None:
            custom_entities._model = custom_entities._model.to("cuda")
    general = a.pipelines["general"]
    model, tok = general.model, general.tokenizer
    id2label = model.config.id2label
    per_ids = [i for i, l in id2label.items() if l.endswith("PER")]
    pack = custom_entities.effective_definitions()

    rows, t0 = [], time.time()
    for k, r in enumerate(sentences()):
        text, s, e = r["text"], r["s"], r["e"]
        # general model as aggregated
        g = best_cover(general(text), s, e)
        # per-token person probability
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            probs = model(**{k2: v.to(dev) for k2, v in enc.items()}).logits[0].softmax(-1).cpu()
        pp = [float(probs[i, per_ids].sum()) for i, (a0, b0) in enumerate(offs) if b0 > a0 and a0 < e and b0 > s]
        # cyber model and GLiNER pack, raw
        cy = overlapping(a.cisco_pipe(text), s, e) if a.cisco_pipe else []
        gl = overlapping(custom_entities.detect(text, pack), s, e)
        # final pipeline
        _, ents, _ = a.anonymize(text, 0)
        f = best_cover(ents, s, e)
        red = set()
        for x in ents:
            if not x.get("preserved"):
                red.update(range(max(x["start"], s), min(x["end"], e)))
        rows.append({**{k2: v for k2, v in r.items() if k2 != "text"},
                     "general_label": g["entity_group"] if g else None,
                     "general_score": float(g["score"]) if g else None,
                     "general_full": bool(g and g["start"] <= s and g["end"] >= e),
                     "p_per_mean": sum(pp) / len(pp) if pp else None,
                     "p_per_min": min(pp) if pp else None,
                     "n_subwords": len(pp),
                     "cyber": [[x["entity_group"], round(float(x["score"]), 3)] for x in cy],
                     "gliner": [[x["entity_group"], round(float(x["score"]), 3)] for x in gl],
                     "final_type": f.get("readable_type") if f else None,
                     "final_score": float(f.get("score", 0) or 0) if f else None,
                     "final_full": bool(f and f["start"] <= s and f["end"] >= e),
                     "leak": len(red) < e - s})
        if k % 2000 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return rows


def pct(n, d):
    return f"{n}/{d} ({n / d:.1%})" if d else "0/0"


def report(rows):
    L = ["# Is the name problem a confidence problem?", "",
         f"{len(rows)} name instances from the two bias tests (Mehrabi 2018 sample, all 4 templates; "
         "origin x gender template test). See name_confidence.py.", ""]

    # 1. the 0.65 cut-off
    per_full = [r for r in rows if r["general_label"] == "PER" and r["general_full"]]
    below = [r for r in per_full if r["general_score"] < CUTOFF]
    L += ["## 1. The 0.65 cut-off on the general model", "",
          f"- Names the general model tagged as a whole person: {len(per_full)}.",
          f"- Of those, scored below {CUTOFF} and so dropped: {pct(len(below), len(per_full))}.",
          f"- Of the dropped ones, leaked in the final output: {pct(sum(r['leak'] for r in below), len(below))}; "
          f"not typed as person: {pct(sum(r['final_type'] != 'PER' for r in below), len(below))}.",
          f"- All leaks: {sum(r['leak'] for r in rows)}; leaks the cut-off could explain: {sum(r['leak'] for r in below)}.", ""]
    L += ["| Test | Gender | Tagged person | Below cut-off | Median score |", "|---|---|---|---|---|"]
    for test in ("mehrabi", "template"):
        for g in "FM":
            c = [r for r in per_full if r["test"] == test and r["gender"] == g]
            sc = sorted(r["general_score"] for r in c)
            L.append(f"| {test} | {g} | {len(c)} | {pct(sum(x < CUTOFF for x in sc), len(c))} | "
                     f"{sc[len(sc) // 2]:.3f} |" if sc else f"| {test} | {g} | 0 | – | – |")
    L.append("")

    # 2. person probability behind wrong labels
    wrong = [r for r in rows if not (r["general_label"] == "PER" and r["general_full"]) and r["p_per_mean"] is not None]
    L += ["## 2. Person probability behind a wrong or missing top label", "",
          f"Names where the general model's top label was not a whole-name person: {len(wrong)}. "
          "Share whose mean per-token person probability reaches each level:", "",
          "| Test | Gender | n | ≥ 0.1 | ≥ 0.2 | ≥ 0.3 | ≥ 0.5 | Median |", "|---|---|---|---|---|---|---|---|"]
    for test in ("mehrabi", "template"):
        for g in "FM":
            c = sorted(r["p_per_mean"] for r in wrong if r["test"] == test and r["gender"] == g)
            if not c:
                continue
            L.append(f"| {test} | {g} | {len(c)} | " + " | ".join(f"{sum(x >= t for x in c) / len(c):.1%}" for t in (0.1, 0.2, 0.3, 0.5))
                     + f" | {c[len(c) // 2]:.3f} |")
    leaked_wrong = [r for r in wrong if r["leak"]]
    L += ["", f"Leaked names among them: {len(leaked_wrong)}; with mean person probability ≥ 0.3: "
          f"{pct(sum(r['p_per_mean'] >= 0.3 for r in leaked_wrong), len(leaked_wrong))}; ≥ 0.5: "
          f"{pct(sum(r['p_per_mean'] >= 0.5 for r in leaked_wrong), len(leaked_wrong))}.",
          "", "What the top label was instead (Mehrabi, template 4):", ""]
    c = {}
    for r in wrong:
        if r["test"] == "mehrabi" and r["group"] == "T4":
            c[r["general_label"] or "(none)"] = c.get(r["general_label"] or "(none)", 0) + 1
    L.append(", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -x[1])))
    L.append("")

    # 3. conflict resolution
    over = [r for r in per_full if r["general_score"] >= CUTOFF and r["final_type"] != "PER"]
    L += ["## 3. Person tags the merge overrode", "",
          f"Names the general model tagged as a person above the cut-off, but the final output typed otherwise: {len(over)} "
          f"(leaked: {sum(r['leak'] for r in over)}).", "",
          "| Final type | n | Median winner score | Median person score | Winner less sure than person tag |",
          "|---|---|---|---|---|"]
    by = {}
    for r in over:
        by.setdefault(r["final_type"] or "(none)", []).append(r)
    for t, c in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ws = sorted(r["final_score"] or 0 for r in c); ps = sorted(r["general_score"] for r in c)
        L.append(f"| {t} | {len(c)} | {ws[len(ws) // 2]:.3f} | {ps[len(ps) // 2]:.3f} | "
                 f"{pct(sum((r['final_score'] or 0) < r['general_score'] for r in c), len(c))} |")
    L.append("")

    # 4. cyber model on names
    apt = [r for r in rows if any(l in ("APT", "MAL") for l, _ in r["cyber"])]
    sc = sorted(max(s for l, s in r["cyber"] if l in ("APT", "MAL")) for r in apt)
    L += ["## 4. Cyber model calling names threat actors or malware", "",
          f"Names with an APT or MAL tag: {len(apt)}; median score {sc[len(sc) // 2]:.3f}; "
          f"≥ 0.8: {pct(sum(x >= 0.8 for x in sc), len(sc))}. Leaked: {sum(r['leak'] for r in apt)}." if apt else "None.", ""]
    (HERE / "NAME_CONFIDENCE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report" in sys.argv and OUT.exists():
        report([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()])
    else:
        report(run())
