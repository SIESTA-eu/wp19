"""
User-defined entity types, found by GLiNER.

The two transformer models in ``base_anonymizer``/``en_anonymizer`` know what
they were trained on: people, organisations and locations from CoNLL-03, and
a cyber-threat vocabulary. That is the right pair for incident reports and
close to useless for anything else. A patient record, a census extract or a
travel database gets names and little else, which is a real limit for anyone
bringing their own data.

GLiNER takes entity types as ordinary words at inference time -- no training,
no fine-tuning -- so a Data Rights Holder can name what matters to them:

    "patient identifier"      -> Patient MRN 4471-B
    "ticket identifier"       -> JIRA-8823
    "vehicle registration"    -> BX21 KLM
    "census ward code"        -> ward E05000123

What the fields are for
-----------------------
``name``        the label sent to the model. This is what does the work, and
                it is worth wording carefully: "patient identifier" finds
                more than "id".
``description`` passed to the model alongside the name, through GLiNER2's
                schema. Measured against label-only on five cases here and
                changed nothing in any of them, so treat it as a hint that
                may help on an ambiguous name rather than something to rely
                on. It costs nothing to send.
``examples``    not sent to the model. They are used to *check* the label:
                run it against the examples and report how many were found,
                so someone can tell a working label from a hopeful one
                before relying on it.
``threshold``   GLiNER returns a confidence per span. Lower finds more and
                is wrong more often; the default suits short factual text.

Cost
----
The model is loaded on first use and only when someone actually has custom
types defined, so a deployment that does not use this pays nothing. It is a
third model in memory when it is loaded; ``GLINER_MODEL`` picks a smaller or
larger one.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("backend")

# GLiNER2 rather than the original: on the cases that matter here it found
# what gliner 0.2 missed -- "MRN 90210", "JIRA-8823", "Respondent 88213",
# a vehicle plate -- all from the label alone. It is a larger encoder
# (deberta-v3-base) and runs at roughly 0.15s for a short document, which is
# fine per document and worth remembering for a long CSV.
DEFAULT_MODEL = "fastino/gliner2-base-v1"
DEFAULT_THRESHOLD = 0.5

# A custom type name becomes an entity type alongside the built-in sixteen,
# so it has to survive being a dict key, a CSV column note and a label in the
# UI. Kept deliberately narrow.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _/-]{1,39}$")

_model = None
_model_failed = False


def model_name() -> str:
    return os.getenv("GLINER_MODEL", DEFAULT_MODEL)


def enabled() -> bool:
    """Whether GLiNER may be used at all -- for the incident-report pack
    below and for owners' own types.

    On by default. It was off until the evaluation against the WP17 models
    showed the two transformer models alone miss most of the personal data
    in incident reports (job titles, ages, usernames, credentials, ID and
    card numbers...), which this model finds from the type name alone. Set
    GLINER_ENABLED=false to run without it, e.g. on very small hosts.
    """
    raw = (os.getenv("GLINER_ENABLED", "true") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# The incident-report pack
# ---------------------------------------------------------------------------
#
# Types every deployment gets without anyone defining them: the personal
# data that cyber incident reports carry and that the two transformer models
# have no label for. Names and descriptions were written from the category
# list of the S-CECILIA annotation guidelines; thresholds were tuned on the
# training portion of that corpus (held-out reports: detection F1 0.68 ->
# 0.71, unredacted identifying characters 22% -> 21%). A threshold below the
# default 0.5 means the type is rarely wrong when it fires and missing it
# costs more; above it, the type over-fires on generic words.
#
# They behave as built-in types: listed in diccionarios.ENTITY_CATALOGUE,
# selectable on the settings page, and redacted unless deselected. An
# owner's own definition with the same name replaces the pack's.
INCIDENT_REPORT_PACK = [
    {"name": "job title", "description": "a person's job title or role, e.g. system administrator", "threshold": 0.3},
    {"name": "age", "description": "a person's age", "threshold": 0.3},
    {"name": "username", "description": "an account username or login handle", "threshold": 0.3},
    {"name": "password", "description": "a password, PIN or other credential", "threshold": 0.5},
    {"name": "identification number", "description": "passport, driver's licence, national ID or employee ID", "threshold": 0.3},
    {"name": "bank or card number", "description": "credit card number, expiry, CVV or bank account number", "threshold": 0.5},
    {"name": "phone number", "description": "a telephone number", "threshold": 0.5},
    {"name": "medical condition", "description": "a health condition, diagnosis or treatment", "threshold": 0.5},
    {"name": "nationality", "description": "a nationality or ethnicity", "threshold": 0.5},
    {"name": "time", "description": "a time of day or timestamp", "threshold": 0.3},
    {"name": "software", "description": "a software product, possibly with a version", "threshold": 0.8},
    {"name": "operating system", "description": "an operating system and version", "threshold": 0.5},
    {"name": "street address", "description": "a street address or postcode", "threshold": 0.5},
    {"name": "amount of money", "description": "a monetary amount", "threshold": 0.6},
    # Counts are quasi-identifiers: "a municipality of 8,000 inhabitants"
    # plus a month and a role can name a town, and so its mayor.
    {"name": "quantity", "description": "a count or amount of people, devices, records or data", "threshold": 0.5},
]


def pack_type(name: str) -> str:
    return name.upper().replace(" ", "_")


PACK_TYPES = [pack_type(d["name"]) for d in INCIDENT_REPORT_PACK]


def effective_definitions(owner_definitions=None) -> list:
    """The pack plus an owner's own types, the owner's winning on a clash.

    Returns [] when GLiNER is disabled, so callers need not check.
    """
    if not enabled():
        return []
    merged = {d["type"]: d for d in (validate(p) for p in INCIDENT_REPORT_PACK)}
    for d in owner_definitions or []:
        merged[d["type"]] = d
    return list(merged.values())


def _load():
    """Load once, and remember a failure rather than retrying per request.

    A missing dependency or an unreachable model hub is a deployment fact,
    not a transient error, and retrying it on every document would turn one
    problem into a slow one.
    """
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    try:
        from gliner2 import GLiNER2
        logger.info("Loading GLiNER model %s (CPU)...", model_name())
        _model = GLiNER2.from_pretrained(model_name())
        logger.info("GLiNER model ready.")
    except Exception as exc:
        _model_failed = True
        logger.warning(
            "Could not load GLiNER (%s). Custom entity types will not be "
            "detected; everything else is unaffected.", exc,
        )
    return _model


def validate(entity: dict) -> dict:
    """Normalise one definition, raising ValueError on anything unusable."""
    name = (entity.get("name") or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            "A type name must start with a letter and be 2-40 characters of "
            "letters, digits, spaces, dash, slash or underscore."
        )

    try:
        threshold = float(entity.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        raise ValueError("Threshold must be a number between 0 and 1.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    examples = [e.strip() for e in (entity.get("examples") or []) if str(e).strip()]

    return {
        "name": name,
        # The type carried through the pipeline. Upper-cased so it sits
        # beside PER/ORG/CVE, and spaced words joined so it survives being
        # written into a CSV note or a settings file.
        "type": name.upper().replace(" ", "_"),
        "description": (entity.get("description") or "").strip()[:500],
        "examples": examples[:20],
        "threshold": threshold,
        "enabled": bool(entity.get("enabled", True)),
    }


def detect(text: str, entities: list[dict]) -> list[dict]:
    """Find the given custom types in ``text``.

    Returns entities in the same shape the rest of the pipeline uses
    (start/end/text/entity_group/score), so ``clean_entities`` can merge them
    with the model and regex findings without knowing where they came from.

    Never raises: a custom type that cannot be resolved should cost its own
    detections, not the whole document.
    """
    active = [e for e in entities if e.get("enabled", True)]
    if not text or not active or not enabled():
        return []

    model = _load()
    if model is None:
        return []

    # One pass with every type, not one per type: the model takes the whole
    # schema at once and this is the expensive part. The description rides
    # along where there is one.
    from gliner2 import Schema

    by_name = {e["name"]: e for e in active}
    schema = Schema().entities({
        e["name"]: (e.get("description") or e["name"]) for e in active
    })
    # The call uses the most permissive threshold so one strict type cannot
    # suppress another's findings; each is re-checked against its own below.
    threshold = min(e.get("threshold", DEFAULT_THRESHOLD) for e in active)

    # Read in overlapping windows: on long reports the model stopped finding
    # anything a few thousand characters in. Duplicates from the overlap
    # keep their higher-confidence copy.
    # format_results=False: gliner2's formatter keeps one span per distinct
    # text, so a nationality or job title named twice in a report was
    # redacted the first time and left in clear the second. The unformatted
    # output keeps every occurrence with its own confidence.
    best = {}
    for a, b in _windows(text):
        try:
            found = model.extract(
                text[a:b], schema, threshold=threshold,
                include_spans=True, include_confidence=True,
                format_results=False,
            )
        except Exception as exc:
            logger.warning("GLiNER prediction failed: %s", exc)
            return []
        entities = found.get("entities") or {}
        if isinstance(entities, list):  # unformatted: one dict per schema part
            entities = {k: v for part in entities for k, v in part.items()}
        for label, spans in entities.items():
            for span in spans or []:
                s, e = a + span["start"], min(a + span["end"], len(text))
                score = span.get("confidence", 0.0)
                if best.get((s, e, label), -1.0) < score:
                    best[(s, e, label)] = score

    out = []
    for (start, end, label), score in sorted(best.items()):
        definition = by_name.get(label)
        if not definition:
            continue
        if score < definition.get("threshold", DEFAULT_THRESHOLD):
            continue
        span = _postprocess(text, definition["name"], start, end)
        if span is None:
            continue
        start, end = span
        entity_type = definition["type"]
        # A "street address" with no number and no street word is a place
        # name (the model called "Aarhus" an address): still redacted, as LOC.
        if definition["name"] == "street address" and not _STREET_WORD.search(text[start:end]) \
                and not re.search(r"\d", text[start:end]):
            entity_type = "LOC"
        out.append({
            "start": start,
            "end": end,
            "text": text[start:end],
            "entity_group": entity_type,
            "score": score,
            # Ranks between the regexes and the models when two sources
            # claim the same span -- see clean_entities.
            "_source": "custom",
        })
    return out


WINDOW_WORDS, WINDOW_OVERLAP = 250, 50


def _windows(text):
    """(start, end) character ranges of overlapping word windows."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if len(words) <= WINDOW_WORDS:
        return [(0, len(text))]
    out = []
    for k in range(0, len(words), WINDOW_WORDS - WINDOW_OVERLAP):
        chunk = words[k:k + WINDOW_WORDS]
        out.append((chunk[0][0], chunk[-1][1]))
        if k + WINDOW_WORDS >= len(words):
            break
    return out


# Post-processing for the pack types, each from an error pattern measured on
# the S-CECILIA training reports. Owners' own types are left as found.
_VALUE_TYPES = {"username", "password", "identification number", "bank or card number", "phone number"}
# A value type fired on the name of the thing rather than a value of it:
# "usernames", "password", "social security number" with no digit in sight.
_CONCEPT_WORDS = re.compile(
    r"\b(user ?names?|passwords?|pins?|credentials?|logins?|social security|driver'?s licen[cs]es?|"
    r"passports?|ids?|identification|numbers?|cards?|credit|expiry|cvvs?|phones?|telephones?|"
    r"accounts?|addresses|emails?)\b", re.IGNORECASE)
_VAGUE_TIME = re.compile(
    r"^(the )?(morning|afternoon|evening|night|midnight|noon|dawn|dusk|sunset|sunrise|today|tonight|"
    r"yesterday|daily|weekly|hours?|minutes?|days?|weeks?|months?|years?)$", re.IGNORECASE)
_GENERIC_SOFTWARE = re.compile(
    r"^(the )?(malware|ransomware|spyware|software|virus|trojan|worm|botnet|antivirus|security software|"
    r"antivirus software|app|apps|application|firmware|code)$", re.IGNORECASE)
_STREET_WORD = re.compile(
    r"\b(street|st|road|rd|avenue|ave|lane|ln|place|pl|boulevard|blvd|drive|dr|way|court|ct|square|sq|plaza|"
    r"terrace|close|crescent|highway|hwy|calle|avenida|plaza|rue|strasse|straße|via|postcode|zip)\b\.?", re.IGNORECASE)
_AGE_BEFORE = re.compile(r"(?:\bage[ds]?\s*:?\s*|\baged\s+)$", re.IGNORECASE)
_AGE_AFTER = re.compile(r"(?:-year-old|\s+years?\s+old|\s*-?\s*years?\b)", re.IGNORECASE)


def _postprocess(text, name, start, end):
    """The span to keep for a pack-type finding, or None to drop it."""
    value = text[start:end].strip()
    if not value:
        return None
    if name in _VALUE_TYPES and not re.search(r"\d", value) and _CONCEPT_WORDS.search(value):
        return None
    if name == "time" and _VAGUE_TIME.match(value):
        return None
    if name == "software" and _GENERIC_SOFTWARE.match(value):
        return None
    if name == "quantity":
        # The model returns "8,000"; the quasi-identifier is "8,000
        # inhabitants". Take in up to four following words that are not
        # function words, stopping at punctuation.
        m = re.match(r"((?:[ -](?!(?:of|in|on|at|to|from|for|and|or|were|was|are|is|by|with|who|that|which)\b)[A-Za-z][\w-]*){1,4})", text[end:])
        if m:
            end += len(m.group(1))
    if name == "age":
        # The model returns "42"; the age is "age 42" or "42-year-old", and
        # leaving the rest behind leaves the reader a template to refill.
        m = _AGE_BEFORE.search(text[max(0, start - 8):start])
        if m:
            start -= len(m.group(0))
        m = _AGE_AFTER.match(text[end:end + 14])
        if m:
            end += len(m.group(0))
    return start, end


# A bland sentence to sit a bare example in. GLiNER reads surrounding words,
# so "Patient MRN 4471-B" on its own is found far less reliably than the same
# value in a sentence -- and people naturally type the bare value.
_CARRIER = "{value} was recorded in the report."


def test_definition(entity: dict) -> dict:
    """Run a definition against its own examples.

    The examples are not training data -- this variant is zero-shot -- so
    this is what they are actually for: telling a label that works from one
    that only sounds right. Reported as found/total with what was matched.
    """
    definition = validate(entity)
    results = []
    for example in definition["examples"]:
        hits = detect(example, [definition])
        if hits:
            results.append({
                "example": example,
                "found": [h["text"] for h in hits],
                "ok": True,
                "needed_context": False,
            })
            continue

        # Nothing found in the example as written. Before calling the label
        # a failure, try the same value inside a sentence: the model reads
        # the words around a span, so a bare value is a harder problem than
        # the real one. Distinguishing the two is the difference between
        # "reword your type name" and "write your examples as sentences",
        # which are different pieces of advice.
        in_context = detect(_CARRIER.format(value=example), [definition])
        results.append({
            "example": example,
            "found": [h["text"] for h in in_context],
            "ok": bool(in_context),
            "needed_context": bool(in_context),
        })
    return {
        "type": definition["type"],
        "name": definition["name"],
        "available": enabled() and _load() is not None,
        "results": results,
        "found": sum(1 for r in results if r["ok"]),
        "needed_context": sum(1 for r in results if r.get("needed_context")),
        "total": len(results),
    }
