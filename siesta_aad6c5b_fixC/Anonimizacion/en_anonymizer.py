from base_anonymizer import BaseAnonymizer, is_plausible_ip, windowed_ner_pipeline
import custom_entities as custom_entities_mod
from logger_config import setup_logger
import diccionarios
import re, os

logger = setup_logger("backend")

# General-purpose English NER model. Replaces
# dbmdz/bert-large-cased-finetuned-conll03-english (334M params) with a
# ~110M-param model that runs comfortably on CPU while covering the same
# PER/ORG/LOC/MISC label set.
GENERAL_NER_MODEL_ID = "dslim/bert-base-NER"
GENERAL_NER_LOCAL_DIR = os.path.join("Models", "dslim_bert-base-NER")


# Small spaCy pipeline, used only for dates and times. The regexes find
# fixed formats; this finds the rest ("the 3rd of May", "early 2023",
# "last Tuesday evening"). Measured on an independent anonymisation benchmark
# (court judgments), it raised recall of quasi-identifiers from 0.71 to 0.77
# at no measurable precision cost. 13 MB, CPU-only.
DATE_MODEL_ID = "en_core_web_sm"

# spaCy's DATE also covers words that carry no date ("daily", "today",
# "zero-day" -- the last is threat intelligence). Keep a span only if it has a
# digit, a month or a weekday in it.
_DATE_ANCHOR = re.compile(
    r"\d|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|"
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b", re.IGNORECASE)


class EnglishAnonymizer(BaseAnonymizer):
    def __init__(self):
        super().__init__()

        model_source = GENERAL_NER_LOCAL_DIR if os.path.exists(GENERAL_NER_LOCAL_DIR) else GENERAL_NER_MODEL_ID

        self.pipelines = {}
        try:
            logger.info(f"Loading general NER model from {model_source} (CPU)...")
            self.pipelines["general"] = windowed_ner_pipeline(model_source)
        except Exception as exc:
            logger.warning(f"Could not load general NER model ({model_source}): {exc}")

        self.date_nlp = None
        try:
            import spacy
            self.date_nlp = spacy.load(DATE_MODEL_ID, exclude=["parser", "lemmatizer", "attribute_ruler", "tagger", "senter"])
        except Exception as exc:
            logger.warning(f"Could not load date model ({DATE_MODEL_ID}): {exc}. Dates rely on the patterns only.")

        # Load GLiNER now rather than on the first request, which would
        # otherwise take the model load (seconds) on top of its own work.
        if custom_entities_mod.enabled():
            custom_entities_mod._load()

    def detect_ner(self, text):
        all_candidates = []
        THRESHOLD = 0.65

        for name, pipe in self.pipelines.items():
            try:
                results = pipe(text)
                for ent in results:
                    if ent['score'] < THRESHOLD:
                        continue

                    raw_label = ent['entity_group'].upper()
                    entity_text = text[ent['start']:ent['end']].strip()

                    # Only accept high-value labels.
                    label = None
                    if any(x in raw_label for x in ["PER", "PERSON"]):
                        label = "PER"
                    elif any(x in raw_label for x in ["ORG", "COMPANY", "GROUP"]):
                        label = "ORG"
                    elif any(x in raw_label for x in ["LOC", "GPE", "CITY", "COUNTRY"]):
                        label = "LOC"
                    elif any(x in raw_label for x in ["DATE", "TIME"]):
                        label = "DATE"
                    elif any(x in raw_label for x in ["MALWARE", "TOOL"]):
                        label = "MALWARE"

                    # Skip labels we don't want and 1-2 character noise.
                    if label and len(entity_text) > 2:
                        all_candidates.append({
                            "start": ent['start'],
                            "end": ent['end'],
                            "text": entity_text,
                            "entity_group": label,
                            "score": ent['score']
                        })
            except Exception as e:
                logger.warning(f"NER inference error ({name}): {e}")
        return all_candidates

    def detect_dates(self, text):
        """Dates and times the fixed patterns do not cover (see DATE_MODEL_ID)."""
        if self.date_nlp is None:
            return []
        out = []
        for ent in self.date_nlp(text).ents:
            if ent.label_ not in ("DATE", "TIME") or not _DATE_ANCHOR.search(ent.text):
                continue
            out.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "text": ent.text,
                "entity_group": ent.label_,
                "score": 0.85,
            })
        return out

    def detect_regex(self, text):
        regex_entities = []
        # Also iterate over the DATE_PATTERNS defined in diccionarios.py.
        for pattern in diccionarios.DATE_PATTERNS:
            for match in pattern.finditer(text):
                regex_entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "entity_group": "DATE",
                    "score": 1.0
                })

        for label, pattern in diccionarios.SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(text):
                if label == "IP_ADDRESS" and not is_plausible_ip(text, match.start(), match.end()):
                    continue
                regex_entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "entity_group": label,
                    "score": 1.0
                })
        return regex_entities

    def anonymize(self, text, option, hash_key=None, redact_types=None, custom_entities=None, scope="adhoc"):
        """``hash_key`` is only consulted for option 4 (hashing) and option 5
        (per-type pseudonymisation), where it is required -- see ``pseudonym``,
        ``pseudonymise`` and ``replace_entities``. ``scope`` is the dataset id
        for option 5: it fixes the date offset and the surrogates.

        ``custom_entities`` are the caller's own type definitions, resolved
        by GLiNER (see ``custom_entities``). They are detected alongside the
        two transformer models and the regexes, and merged by the same
        overlap rules, so a custom type competing with a built-in one is
        settled the same way as any other clash. Whether they are then
        replaced is still ``redact_types``' decision.

        ``redact_types`` is the set of entity types to actually replace, as
        they appear in ``diccionarios.ENTITY_CATALOGUE``. Everything else is
        still detected and still reported in the returned entity list, just
        left in the text. Passing None keeps the deployment-wide default
        (everything except ``diccionarios.preserved_types()``), which is
        what the ad-hoc endpoints use; the dataset pipeline passes the
        owner's own selection.

        Detection is never narrowed by this. A type nobody wants redacted is
        still found and still reported, so the owner can see what was left
        in and change their mind -- and so the audit trail records what was
        in the text either way.
        """
        processed_text = self.replace_predefined_acronyms(text)
        ner_ents = self.detect_ner(processed_text)
        cyber_ents = self.detect_cyber_entities(processed_text) + self.detect_dates(processed_text)
        reg_ents = self.detect_regex(processed_text)

        # The incident-report pack plus the caller's own types (theirs win on
        # a name clash); empty when GLiNER is disabled. Grouped with the
        # regexes rather than the models: both are explicit statements of
        # what to look for, and on an identical span clean_entities prefers
        # them to a model's guess.
        definitions = custom_entities_mod.effective_definitions(custom_entities)
        if definitions:
            reg_ents = reg_ents + custom_entities_mod.detect(processed_text, definitions)

        cleaned = self.clean_entities(processed_text, ner_ents + cyber_ents, reg_ents)

        # Everything detected is reported; only the chosen types are
        # rewritten. See diccionarios.preserved_types for why the default
        # leaves CVE ids, malware names and file hashes alone -- they make
        # an incident report useful to the next reader and identify no one.
        if redact_types is not None:
            wanted = {t.upper() for t in redact_types}
            for e in cleaned:
                e['preserved'] = (e.get('readable_type') or '').upper() not in wanted
        redactable = [e for e in cleaned if not e.get('preserved')]

        if option == 1:
            mod = self.replace_entities(processed_text, redactable, mode="label")
        elif option == 2:
            mod = self.replace_entities(processed_text, redactable, mode="mask")
        elif option == 3:
            mod = self.replace_with_fake_data(processed_text, redactable)
        elif option == 4:
            mod = self.replace_entities(processed_text, redactable, mode="hash", hash_key=hash_key)
        elif option == 5:
            mod = self.replace_entities(processed_text, redactable, mode="pseudonymise",
                                        hash_key=hash_key, scope=scope)
        else:
            mod = processed_text

        return processed_text, cleaned, mod