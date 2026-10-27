"""SPY (Synthetic PII Yesterday) through the SIESTA anonymiser, before and
after the whole-word fix.

SPY (third_party/spy_src, LogicZMaksimka/SPY_Dataset commit 5baabd6, data
only) ships 4,197 legal questions and 4,491 medical consultations with
placeholders (<NAME>, <EMAIL>, ...) that its loader fills with Faker values.
The filling is reproduced here from spy.py and src/utils.py rather than run:
Faker() (en_US) seeded 0, one generated profile per document, each
generator's share split evenly and shuffled. Their shuffle uses Python's
unseeded `random`, and Faker's output differs between versions, so values
are not identical to any other run of SPY; here `random` is seeded 0 too.

Each document goes through the release pipeline twice, as in
entity_switch_bias.py: general model "simple" (before the fix) and "first"
(after), the other detectors shared.

Per gold entity, by SPY type:
  redacted  fully covered by redacted (non-preserved) entities
  typed     covered by one entity of the matching SIESTA type
and, over all types, type-agnostic detection P/R/F1 at IoU >= 0.5. SIESTA
also detects types SPY does not annotate (dates, job titles, quantities...);
those count as false positives, so precision understates it. Names are also
split by gender, from Faker's en_US female/male first-name lists.

Glued values: SPY's placeholder tokens are flagged as having no trailing
whitespace, and the token after them often carries none in front, so the
filled value is joined to the next word ("559-845-7499x96104to discuss").
SPY's loader produces the same text. Gold values touching a letter or digit
on either side are excluded from the per-type and gender tables (counted in
the "glued" column); the type-agnostic P/R/F1 keeps every value.

Usage: [SIESTA_DEVICE=0] spy_benchmark.py [--report]
"""
import json, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import entity_switch_bias as esb   # build(): the shared two-variant anonymiser

SPY = HERE / "third_party" / "spy_src" / "repo" / "data"
OUT = HERE / "logs" / "spy_benchmark.jsonl"
DOMAINS = {"legal": "legal_questions_placeholders.jsonl", "medical": "medical_consultations_placeholders.jsonl"}
VARIANTS = esb.VARIANTS
TYPE_MAP = {"NAME": {"PER"}, "EMAIL": {"EMAIL"}, "URL": {"URL", "DOMAIN"}, "PHONE_NUM": {"PHONE_NUMBER"},
            "ID_NUM": {"IDENTIFICATION_NUMBER", "BANK_OR_CARD_NUMBER"}, "ADDRESS": {"STREET_ADDRESS", "LOC"},
            "USERNAME": {"USERNAME"}}


def faker_funcs(faker):
    """PII_ENT_FUNCS from spy.py."""
    return {
        "EMAIL": [faker.ascii_email, faker.ascii_free_email],
        "NAME": [faker.name],
        "URL": [lambda: faker.uri(deep=1), lambda: faker.uri(deep=2)],
        "PHONE_NUM": [faker.phone_number],
        "ID_NUM": [faker.ripe_id, faker.msisdn, faker.ssn, faker.sbn9, faker.isbn10, faker.isbn13,
                   faker.credit_card_number, faker.aba, faker.bban, faker.iban],
        "ADDRESS": [faker.street_address],
        "USERNAME": [faker.user_name],
    }


def profiles(n):
    """generate_profiles() from src/utils.py, seeded."""
    from faker import Faker
    Faker.seed(0); random.seed(0)
    f = Faker()
    out = {}
    for ty, funcs in faker_funcs(f).items():
        sizes = [n // len(funcs) + (i < n % len(funcs)) for i in range(len(funcs))]
        vals = [fn() for fn, k in zip(funcs, sizes) for _ in range(k)]
        random.shuffle(vals)
        out[ty] = vals
    return out


def documents():
    for dom, fn in DOMAINS.items():
        rows = [json.loads(l) for l in open(SPY / fn, encoding="utf-8")]
        prof = profiles(len(rows))
        for i, r in enumerate(rows):
            text, gold = "", []
            for tok, tag, ws in zip(r["tokens"], r["ent_tags"], r["trailing_whitespaces"]):
                if tag.startswith("B-"):
                    ty = tag[2:]
                    words = prof[ty][i].split() if ty in prof else [tok]
                    s = len(text)
                    for j, w in enumerate(words):
                        text += w
                        if j < len(words) - 1 or ws:
                            text += " "
                    gold.append((s, s + len(" ".join(words)), ty))
                elif tag.startswith("I-"):
                    text += tok   # placeholders are single tokens; kept for safety
                else:
                    text += tok
                    if ws and not tok.endswith((" ", "\n")):
                        text += " "
            yield dom, i, text, gold


def run():
    a, pipes, cache = esb.build()
    t0 = time.time()
    with open(OUT.with_suffix(".part"), "w", encoding="utf-8") as f:
        for k, (dom, i, text, gold) in enumerate(documents()):
            rec = {"domain": dom, "i": i, "gold": gold, "names": [text[s:e] for s, e, t in gold if t == "NAME"]}
            cache.clear()
            for v in VARIANTS:
                a.pipelines["general"] = pipes[v]
                _, ents, _ = a.anonymize(text, 0)
                rec[v] = [(x["start"], x["end"], x.get("readable_type"), bool(x.get("preserved"))) for x in ents]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 500 == 0:
                print(dom, i, f"{time.time() - t0:.0f}s", flush=True)
    OUT.with_suffix(".part").rename(OUT)


def outcomes(rec, v):
    out = []
    for s, e, ty in rec["gold"]:
        red = set()
        for a_, b, _, pres in rec[v]:
            if not pres:
                red.update(range(max(a_, s), min(b, e)))
        typed = any(t in TYPE_MAP[ty] and a_ <= s and b >= e for a_, b, t, _ in rec[v])
        out.append((ty, len(red) >= e - s, typed, s, e))
    return out


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def detect_prf(recs, v):
    tp = np_ = ng = 0
    for r in recs:
        pred = [(a_, b) for a_, b, _, _ in r[v]]
        gold = [(s, e) for s, e, _ in r["gold"]]
        used = set()
        for g in gold:
            best = max(((iou(g, p), j) for j, p in enumerate(pred) if j not in used), default=(0, None))
            if best[0] >= 0.5:
                used.add(best[1]); tp += 1
        np_ += len(pred); ng += len(gold)
    p, rc = tp / max(np_, 1), tp / max(ng, 1)
    return p, rc, 2 * p * rc / max(p + rc, 1e-9)


def glued(text, s, e):
    return (e < len(text) and text[e].isalnum()) or (s > 0 and text[s - 1].isalnum())


def drop_glued(recs):
    """Mark each gold value of each record as glued to a neighbouring word."""
    texts = {(dom, i): text for dom, i, text, _ in documents()}
    for r in recs:
        text = texts[(r["domain"], r["i"])]
        r["glued"] = [glued(text, s, e) for s, e, _ in r["gold"]]


def clean(rec, v):
    return [x for x, g in zip(outcomes(rec, v), rec["glued"]) if not g]


def report(recs):
    from faker.providers.person.en_US import Provider as P
    drop_glued(recs)
    fem, mal = set(P.first_names_female), set(P.first_names_male)
    L = ["# SPY through SIESTA, before and after the whole-word fix", "",
         f"{len(recs)} documents (legal questions, medical consultations); Faker en_US values, seed 0. "
         "`simple` = before the fix, `first` = after.", "",
         f"Gold values glued to a neighbouring word by SPY's text assembly are excluded from the first two tables: "
         f"{sum(sum(r['glued']) for r in recs)} of {sum(len(r['glued']) for r in recs)}.", ""]
    L += ["## Per entity type", "", "| Domain | SPY type | n | glued (excluded) | Redacted (simple → first) | Right type (simple → first) |", "|---|---|---|---|---|---|"]
    for dom in DOMAINS:
        rs = [r for r in recs if r["domain"] == dom]
        for ty in TYPE_MAP:
            o = {v: [x for r in rs for x in clean(r, v) if x[0] == ty] for v in VARIANTS}
            n = len(o["simple"])
            ng = sum(g for r in rs for (_, _, t), g in zip(r["gold"], r["glued"]) if t == ty)
            red = " → ".join(f"{sum(x[1] for x in o[v]) / n:.1%}" for v in VARIANTS)
            typ = " → ".join(f"{sum(x[2] for x in o[v]) / n:.1%}" for v in VARIANTS)
            L.append(f"| {dom} | {ty} | {n} | {ng} | {red} | {typ} |")
    L += ["", "## Names by gender (Faker en_US first-name lists)", "", "| Gender | n | Redacted (simple → first) | Typed PER (simple → first) |", "|---|---|---|---|"]
    for g, names in (("F", fem), ("M", mal)):
        sel = {v: [] for v in VARIANTS}
        for r in recs:
            for v in VARIANTS:
                names_g = [g for (_, _, t), g in zip(r["gold"], r["glued"]) if t == "NAME"]
                for (ty, red, typed, s, e), nm, gl in zip([x for x in outcomes(r, v) if x[0] == "NAME"], r["names"], names_g):
                    if gl: continue
                    first = [w for w in nm.split() if w.rstrip(".") not in {"Mr", "Mrs", "Ms", "Miss", "Dr"}][:1]
                    if first and first[0] in names and first[0] not in (fem & mal):
                        sel[v].append((red, typed))
        n = len(sel["simple"])
        L.append(f"| {g} | {n} | " + " → ".join(f"{sum(x[0] for x in sel[v]) / max(n, 1):.1%}" for v in VARIANTS)
                 + " | " + " → ".join(f"{sum(x[1] for x in sel[v]) / max(n, 1):.1%}" for v in VARIANTS) + " |")
    L += ["", "## Type-agnostic detection (IoU ≥ 0.5)", "", "| Domain | P / R / F1, simple | P / R / F1, first |", "|---|---|---|"]
    for dom in DOMAINS:
        rs = [r for r in recs if r["domain"] == dom]
        L.append(f"| {dom} | " + " | ".join("{:.3f} / {:.3f} / {:.3f}".format(*detect_prf(rs, v)) for v in VARIANTS) + " |")
    L.append("\nPrecision counts every SIESTA detection SPY does not annotate as a false positive: types it does not cover "
             "(dates, job titles, quantities...) and identifiers of its own types that the generated text contains outside "
             "the placeholders (e-mail addresses in about 95% of documents, titled names in 97% of medical ones).")
    (HERE / "SPY_BENCHMARK.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report" in sys.argv and OUT.exists():
        report([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()])
    else:
        run()
        report([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()])
