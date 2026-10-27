"""SPY (Synthetic PII Yesterday) documents and scoring, for run_spy_variants.py.

SPY (third_party/spy_src, LogicZMaksimka/SPY_Dataset commit 5baabd6, data
only) ships 4,197 legal questions and 4,491 medical consultations with
placeholders (<NAME>, <EMAIL>, ...) that its loader fills with Faker values.
The filling is reproduced here from spy.py and src/utils.py rather than run:
Faker() (en_US) seeded 0, one generated profile per document, each
generator's share split evenly and shuffled. Their shuffle uses Python's
unseeded `random`, and Faker's output differs between versions, so values
are not identical to any other run of SPY; here `random` is seeded 0 too.

Per gold entity, by SPY type:
  redacted  fully covered by redacted (non-preserved) entities
  typed     covered by one entity of the matching SIESTA type
and, over all types, type-agnostic detection P/R/F1 at IoU >= 0.5. SIESTA
also detects types SPY does not annotate (dates, job titles, quantities...);
those count as false positives, so precision understates it.

Glued values: SPY's placeholder tokens are flagged as having no trailing
whitespace, and the token after them often carries none in front, so the
filled value is joined to the next word ("559-845-7499x96104to discuss").
SPY's loader produces the same text. Gold values touching a letter or digit
on either side are excluded from the per-type table (drop_glued); the
type-agnostic P/R/F1 keeps every value.
"""
import json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)

SPY = ROOT / "third_party" / "spy_src" / "repo" / "data"
DOMAINS = {"legal": "legal_questions_placeholders.jsonl", "medical": "medical_consultations_placeholders.jsonl"}
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
