"""SIESTA anonymiser (uc4_text_anonymization, commit 0a358bf) on every gold report.

In D19.4: `run_siesta.py aad-fixC` is "RubiNER without S-CECILIA fine-tuning"
(rubiner/anonymiser/ as it stands), and run_rubiner.py builds RubiNER on top
of it. The other variants below are earlier experiments, not in D19.4.

Variants
  siesta          exactly as deployed: dslim/bert-base-NER + SecureBERT-NER
                  + regexes, CPU, no custom types. The HF pipelines are
                  built without a stride, so each model reads only the first
                  512 tokens of a report.
  siesta-stride   the same, with both pipelines rebuilt with stride=128 so
                  the whole report is read. Patched here, not in the repo.
  siesta-sb2      siesta-stride with the cyber model swapped for
                  cisco-ai/SecureBERT2.0-NER (ModernBERT, 8k context; labels
                  Indicator / Malware / Organization / System / Vulnerability
                  mapped into CYBER_LABEL_MAP in memory)
  siesta-sb2-gliner  siesta-gliner with the same swap
  siesta-head     the repo as it stands (commit aad6c5b), unpatched: GLiNER2
                  pack on, default configuration
  siesta-head-first  siesta-head with the general model's pipeline rebuilt
                  with aggregation_strategy="first", so each whole word gets
                  the label of its first sub-word token. The "simple"
                  strategy tags only the first piece of a name it splits
                  ("Be" of "Bea"), and detect_ner then drops it as 1-2
                  character noise before spans are snapped to words.
                  Patched here, not in the repo.
  siesta-wt       the repo's working tree as it stands (aad6c5b + the general
                  model on aggregation_strategy="first"), unpatched
  siesta-fix5     the working tree with fix 5 (GLiNER2 person arbiter;
                  confidence breaks custom/model ties), fail-closed off and
                  no cyber threshold (base_anonymizer constants set here)
  siesta-fix56    plus fix 6: fail closed on names (FAIL_CLOSED_NAMES)
  siesta-fix567-t<x>  plus fix 7: cyber-model findings below score x dropped
                  (CYBER_MIN_SCORE = x)
  siesta-fix57-t<x>   fixes 5 and 7 without fix 6 (the configuration chosen
                  on 13 October: fix 6 became a stated limitation)
  abl-<part>      leave-one-out ablation of siesta-wt: the working tree with
                  one component removed at run time --
                  abl-no-general (general NER), abl-no-cyber (cyber NER),
                  abl-no-gliner (incident-report pack), abl-no-dates (spaCy
                  dates), abl-no-patterns (regular expressions, date patterns
                  included), abl-no-namecomp (completion of partial names);
                  abl-general-only keeps the general model alone (plus the
                  merge), the floor the other parts are added to
  siesta-gliner   siesta-stride plus GLiNER2 custom entity types, defined
                  the way a Data Rights Holder would through the settings
                  page: for the kinds of personal data the built-in sixteen
                  types do not cover. Names were written from the S-CECILIA
                  guidelines' category list, not tuned on the gold.

Detection is scored, not redaction: every entity the anonymiser reports
counts, including those it would preserve (CVE, MALWARE, HASH).

Offsets: anonymize() runs acronym expansion first, so its offsets refer to
the expanded text. Where expansion changed a report, it is re-run with
expansion disabled so spans line up with the gold (counted in the meta).
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
REPO = ROOT / "rubiner" / "anonymiser"   # the anonymiser evaluated in D19.4
# SIESTA_REPO points the harness at another copy of the code.
if os.environ.get("SIESTA_REPO"):
    REPO = Path(os.environ["SIESTA_REPO"])
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(REPO))
os.chdir(REPO)
import corpus  # noqa: E402

CUSTOM_TYPES = [
    {"name": "job title", "description": "a person's job title or role, e.g. system administrator"},
    {"name": "age", "description": "a person's age"},
    {"name": "username", "description": "an account username or login handle"},
    {"name": "password", "description": "a password, PIN or other credential"},
    {"name": "identification number", "description": "passport, driver's licence, national ID or employee ID"},
    {"name": "bank or card number", "description": "credit card number, expiry, CVV or bank account number"},
    {"name": "phone number", "description": "a telephone number"},
    {"name": "medical condition", "description": "a health condition, diagnosis or treatment"},
    {"name": "nationality", "description": "a nationality or ethnicity"},
    {"name": "time", "description": "a time of day or timestamp"},
    {"name": "software", "description": "a software product, possibly with a version"},
    {"name": "operating system", "description": "an operating system and version"},
    {"name": "street address", "description": "a street address or postcode"},
    {"name": "amount of money", "description": "a monetary amount"},
]


SB2 = "cisco-ai/SecureBERT2.0-NER"
SB2_LABELS = {"ORGANIZATION": "ORG", "SYSTEM": "TECH", "INDICATOR": "TECH"}  # MALWARE, VULNERABILITY already mapped


# SIESTA_DEVICE=0 runs the models on the GPU to save time; detection is the
# same up to floating-point noise. Default -1 (CPU), as deployed.
DEV = int(os.environ.get("SIESTA_DEVICE", "-1"))


def patch_general_first(a):
    """Rebuild the general model's pipeline as windowed_ner_pipeline does,
    but with aggregation_strategy="first" (siesta-head-first)."""
    from transformers import pipeline
    import base_anonymizer
    old = a.pipelines["general"]
    a.pipelines["general"] = pipeline(
        "ner", model=old.model, tokenizer=old.tokenizer, aggregation_strategy="first",
        stride=base_anonymizer.NER_STRIDE, device=-1)


ABLATIONS = ("abl-no-general", "abl-no-cyber", "abl-no-gliner", "abl-no-dates",
             "abl-no-patterns", "abl-no-namecomp", "abl-general-only")


def ablate(a, variant):  # variant without any "aad-" prefix
    """Remove one component of the working tree at run time (abl-*).
    The pack is switched off in main(), through GLINER_ENABLED, as an
    operator would."""
    import base_anonymizer
    if variant in ("abl-no-general",):
        a.pipelines.pop("general", None)
    if variant in ("abl-no-cyber", "abl-general-only"):
        a.cisco_pipe = None
    if variant in ("abl-no-dates", "abl-general-only"):
        a.date_nlp = None
    if variant in ("abl-no-patterns", "abl-general-only"):
        a.detect_regex = lambda text: []
    if variant == "abl-no-namecomp":
        base_anonymizer.BaseAnonymizer._extend_names = classmethod(lambda cls, text, entities: entities)


def build(variant):
    from en_anonymizer import EnglishAnonymizer
    a = EnglishAnonymizer()
    if variant.startswith(("siesta-v", "siesta-head", "siesta-wt", "abl-", "siesta-fix", "aad-")):
        # This release exactly as committed: no patches. Only moved to the
        # GPU for speed when asked (same output, checked span for span).
        if variant == "siesta-head-first":
            patch_general_first(a)
        if variant.removeprefix("aad-").startswith("abl-"):
            ablate(a, variant.removeprefix("aad-"))
        if variant.startswith("siesta-fix"):
            import base_anonymizer
            base_anonymizer.FAIL_CLOSED_NAMES = "6" in variant.split("-")[1]
            base_anonymizer.CYBER_MIN_SCORE = float(variant.split("-t")[1]) if "-t" in variant else None
        if DEV >= 0:
            import torch
            for pipe in list(a.pipelines.values()) + ([a.cisco_pipe] if a.cisco_pipe else []):
                pipe.model.to("cuda"); pipe.device = torch.device("cuda")
        return a
    if variant == "siesta" and DEV >= 0:
        from transformers import pipeline
        for key, pipe in list(a.pipelines.items()):
            a.pipelines[key] = pipeline("ner", model=pipe.model, tokenizer=pipe.tokenizer,
                                        aggregation_strategy="simple", device=DEV)
        if a.cisco_pipe is not None:
            p = a.cisco_pipe
            a.cisco_pipe = pipeline("ner", model=p.model, tokenizer=p.tokenizer, aggregation_strategy="simple", device=DEV)
    if variant != "siesta":
        from transformers import pipeline
        for key, pipe in list(a.pipelines.items()):
            pipe.tokenizer.model_max_length = 512
            a.pipelines[key] = pipeline("ner", model=pipe.model, tokenizer=pipe.tokenizer,
                                        aggregation_strategy="simple", stride=128, device=DEV)
        if "sb2" in variant:
            import base_anonymizer
            base_anonymizer.CYBER_LABEL_MAP.update(SB2_LABELS)
            raw = pipeline("ner", model=SB2, aggregation_strategy="simple", device=DEV)

            def trimmed(text):
                # ModernBERT offsets include the leading space (" Green"), and
                # SIESTA's word-boundary snap then walks through it into the
                # previous word ("at GreenTech"). Trim before handing over.
                out = []
                for e in raw(text):
                    st, en = e["start"], e["end"]
                    while st < en and text[st].isspace(): st += 1
                    while en > st and text[en - 1].isspace(): en -= 1
                    if en > st: out.append({**e, "start": st, "end": en})
                return out
            a.cisco_pipe = trimmed
        elif a.cisco_pipe is not None:
            p = a.cisco_pipe; p.tokenizer.model_max_length = 512
            a.cisco_pipe = pipeline("ner", model=p.model, tokenizer=p.tokenizer,
                                    aggregation_strategy="simple", stride=128, device=DEV)
    return a


def main(variant):
    gl = variant.endswith("gliner")
    if variant.startswith(("siesta-v", "siesta-head", "siesta-wt", "abl-", "siesta-fix", "aad-")):
        # GLiNER on by default in this release; the pack comes from the repo.
        os.environ.pop("GLINER_ENABLED", None)
    if gl:
        os.environ["GLINER_ENABLED"] = "true"
    if variant.removeprefix("aad-") in ("abl-no-gliner", "abl-general-only"):
        os.environ["GLINER_ENABLED"] = "false"
    import custom_entities
    customs = [custom_entities.validate(c) for c in CUSTOM_TYPES] if gl else None
    if variant == "siesta-v3-noqty":
        # the same release with the quantity type switched off, through the
        # owner override the product offers
        customs = [custom_entities.validate({"name": "quantity", "enabled": False})]
    if (gl or variant.startswith(("siesta-v", "siesta-head", "siesta-wt", "abl-", "siesta-fix", "aad-"))) and DEV >= 0 and custom_entities._load() is not None:
        custom_entities._model = custom_entities._model.to("cuda")
    a = build(variant)
    identity = lambda text: text
    docs = corpus.load()
    out = corpus.preds_dir() / f"{variant}.jsonl"
    part = out.with_suffix(".part")
    rerun = 0; t0 = time.time()
    with open(part, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            processed, ents, _ = a.anonymize(d["text"], 0, custom_entities=customs)
            if processed != d["text"]:
                rerun += 1
                orig = a.replace_predefined_acronyms
                a.replace_predefined_acronyms = identity
                processed, ents, _ = a.anonymize(d["text"], 0, custom_entities=customs)
                a.replace_predefined_acronyms = orig
            f.write(json.dumps({"id": d["id"], "spans": [[e["start"], e["end"], e["entity_group"]] for e in ents]}) + "\n")
            if i % 100 == 0: print(variant, i, f"{time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    meta = {"_meta": {"name": {"siesta": "SIESTA (as deployed)", "siesta-stride": "SIESTA + stride fix",
                               "siesta-gliner": "SIESTA + stride + GLiNER2 custom types",
                               "siesta-sb2": "SIESTA + SecureBERT 2.0",
                               "siesta-sb2-gliner": "SIESTA + SecureBERT 2.0 + GLiNER2 custom types",
                               "siesta-v2": "SIESTA v2 (release eefa7a9)",
                               "siesta-v3": "SIESTA v2.1 (fixes + quantity)",
                               "siesta-v3-noqty": "SIESTA v2.1 without quantity",
                               "siesta-head": "SIESTA aad6c5b",
                               "siesta-head-first": "SIESTA aad6c5b, general model aggregation=first",
                               "siesta-wt": "SIESTA aad6c5b + whole-word fix (working tree)"}.get(variant, variant),
                      "taxonomy": "native", "family": "SIESTA", "seconds": round(secs, 1), "device": "cpu" if DEV < 0 else "cuda",
                      "acronym_reruns": rerun}}
    with open(out, "w", encoding="utf-8") as g:
        g.write(json.dumps(meta) + "\n"); g.write(part.read_text(encoding="utf-8"))
    part.unlink()
    print(variant, "done", f"{secs:.0f}s", "acronym reruns", rerun, flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["siesta", "siesta-stride", "siesta-gliner"]:
        main(v)
