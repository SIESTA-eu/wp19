import re
import logging
import diccionarios
import pseudonym
import pseudonymise
from diccionarios import SENSITIVE_PATTERNS
from transformers import pipeline
import os

from logger_config import setup_logger

logger = setup_logger("backend")

# Cyber-threat-intel NER model, fine-tuned for exactly this task (malware,
# threat actors, CVEs, IOCs). Replaces the old "cisco-ai_SecureBERT2.0-ner"
# reference, which pointed at a folder that was never vendored
# ("cisco-ai_SecureBERT2.0-base" is a base encoder with no NER head) --
# self.cisco_pipe was silently None before this change.
CYBER_NER_MODEL_ID = "CyberPeace-Institute/SecureBERT-NER"
CYBER_NER_LOCAL_DIR = os.path.join("Models", "CyberPeace-Institute_SecureBERT-NER")

# Maps the cyber model's raw entity labels onto our internal taxonomy.
# Unrecognised labels fall back to "TECH" (see detect_cyber_entities) rather
# than crashing, since the exact label set is only known once the model's
# config.json has actually been downloaded.
CYBER_LABEL_MAP = {
    # ACT is the model's "attack action" label (APTNER): techniques such as
    # "phishing" or "brute-force". It names no one; it used to be mapped to
    # PER, so every "phishing" was redacted as a person. Not mapped: see
    # CYBER_SKIP_LABELS.
    "APT": "MALWARE",
    "SECTEAM": "ORG", "IDTY": "ORG", "IDENTITY": "ORG",
    "OS": "TECH", "EMAIL": "EMAIL", "LOC": "LOC",
    "PER": "PER", "PERSON": "PER",
    "IP": "IP", "IPV4": "IP", "IPV6": "IP",
    "DOM": "DOMAIN", "DOMAIN": "DOMAIN", "URL": "URL",
    "PROT": "TECH", "PROTOCOL": "TECH",
    "FILE": "FILE", "FILEPATH": "FILE_PATH",
    "TOOL": "TECH",
    "MD5": "HASH", "SHA1": "HASH", "SHA2": "HASH", "HASH": "HASH",
    "MAL": "MALWARE", "MALWARE": "MALWARE",
    "ENCR": "TECH", "ENCRYPTION": "TECH",
    "VULNAME": "CVE", "VULNERABILITY": "CVE", "VULID": "CVE", "CVE": "CVE",
    "TIME": "DATE", "DATE": "DATE",
}


# Both transformer models read at most 512 tokens at a time. Without a stride
# the pipeline stops there and says nothing: everything after roughly the
# first 2,000 characters of a report went through undetected, so a name in
# the second half of a long report was never redacted. With a stride the
# text is read in overlapping 512-token windows and the results are joined.
NER_MAX_TOKENS = 512
NER_STRIDE = 128


def windowed_ner_pipeline(model_source):
    """A token-classification pipeline that reads the whole text, not just
    its first NER_MAX_TOKENS tokens."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    # Some tokenizers ship without a usable limit (a huge sentinel), which
    # makes the stride a no-op; the models themselves stop at 512.
    tokenizer.model_max_length = NER_MAX_TOKENS
    return pipeline(
        "ner",
        model=model_source,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        stride=NER_STRIDE,
        device=-1,
    )


# IP addresses are checked, not just matched. In scanner output the pattern
# alone fires on SNMP OIDs (the "1.3.6.1" of 1.3.6.1.4.1.25623.1.0.103692),
# dotted version numbers ("version 4.1.2.3") and IPv6-looking fragments
# ("9390::"), and the cyber model labels plain decimals ("4.1", "0.105782")
# as IP. Redacted, every version number in a vulnerability report became an
# "IP address" and the report lost its technical content.
_VERSION_CONTEXT = re.compile(r"(?:\bversion|\bver\.?|\bbuild|\brelease|\bv)\s*$", re.IGNORECASE)


def is_plausible_ip(text, start, end):
    """Whether text[start:end] is an IP address in its context."""
    import ipaddress
    candidate = text[start:end]
    # Part of a longer dotted number: an OID, or an address-looking prefix
    # of one ("1.3.6.1" followed by ".4.1...").
    if start > 0 and (text[start - 1].isdigit() or (text[start - 1] == "." and start > 1 and text[start - 2].isdigit())):
        return False
    if end < len(text) - 1 and text[end] == "." and text[end + 1].isdigit():
        return False
    if _VERSION_CONTEXT.search(text[max(0, start - 12):start]):
        return False
    bare = candidate.split("/")[0]                    # CIDR suffix
    if bare.count(":") == 1 and "." in bare:          # IPv4:port
        bare = bare.split(":")[0]
    if bare.startswith("[") and "]" in bare:          # [IPv6]:port
        bare = bare[1:bare.index("]")]
    try:
        ipaddress.ip_address(bare)
    except ValueError:
        return False
    return True


# Cyber-model labels that are not identifiers and are dropped.
CYBER_SKIP_LABELS = {"ACT"}
# Labels that should be names. The model also puts them on generic nouns
# ("attacker", "employees", "ransomware", "exploit" as a vulnerability name,
# "email" as a tool); a name is capitalised or has a digit, so an
# all-lower-case span under one of these labels is dropped. For TOOL this
# also drops lower-case tool names ("mimikatz"); on the S-CECILIA training
# reports that cost no private leak and removed hundreds of false "email"
# detections.
CYBER_NAME_LABELS = {"APT", "IDTY", "SECTEAM", "MAL", "VULNAME", "VULID", "TOOL"}


class BaseAnonymizer:
    def __init__(self):
        self.patterns = SENSITIVE_PATTERNS
        self.cisco_pipe = self._load_cyber_pipeline()

    def _load_cyber_pipeline(self):
        """Load the cyber-threat-intel NER model, CPU-only.

        Prefers a locally vendored copy under Models/ (for air-gapped
        deployments); otherwise loads straight from the Hugging Face Hub id,
        which transformers downloads and caches on first use.
        """
        model_source = CYBER_NER_LOCAL_DIR if os.path.exists(CYBER_NER_LOCAL_DIR) else CYBER_NER_MODEL_ID
        try:
            logger.info(f"Loading cyber-entity NER model from {model_source} (CPU)...")
            return windowed_ner_pipeline(model_source)
        except Exception as exc:
            logger.warning(f"Could not load cyber-entity NER model ({model_source}): {exc}")
            return None

    def detect_cyber_entities(self, text):
        """Detect cyber-threat-intel entities (malware, actors, CVEs, IOCs)."""
        if not self.cisco_pipe:
            return []

        cyber_results = self.cisco_pipe(text)
        cyber_entities = []

        for ent in cyber_results:
            # Strip any B-/I- prefix so the mapping lookup works.
            raw_label = ent['entity_group'].upper()
            clean_label = raw_label.split('-')[-1]

            if clean_label in CYBER_SKIP_LABELS:
                continue
            span_text = text[ent['start']:ent['end']].strip()
            if clean_label in CYBER_NAME_LABELS and span_text == span_text.lower() and not any(c.isdigit() for c in span_text):
                continue
            label = CYBER_LABEL_MAP.get(clean_label, "TECH")
            if clean_label not in CYBER_LABEL_MAP:
                logger.info(f"Unmapped cyber NER label '{clean_label}' -> defaulting to TECH")
            # The model calls version numbers and scores "IP"; keep the label
            # only for text that actually is an address.
            if label == "IP" and not is_plausible_ip(text, ent['start'], ent['end']):
                continue

            cyber_entities.append({
                "start": ent['start'],
                "end": ent['end'],
                "text": text[ent['start']:ent['end']],
                "entity_group": label,
                "score": ent['score']
            })
        return cyber_entities

    def replace_predefined_acronyms(self, text):
        """Off unless ACRONYM_EXPANSION=true.

        Expansion rewrote the owner's text: the anonymised output held
        descriptions the author never wrote, threat-actor names turned into
        descriptions ("BlackCat" -> "Notorious ransomware group", which the
        malware patterns then no longer matched), and context-free entries
        inserted false facts ("EPS" -> "Colombian health insurance
        provider"). The list was built from one earlier corpus. Left in place
        for anyone relying on it; the right design, if it comes back, is to
        expand for detection only and replace in the original wording.
        """
        if (os.getenv("ACRONYM_EXPANSION", "") or "").strip().lower() not in {"1", "true", "yes", "on"}:
            return text
        for acronym, expansion in diccionarios.predefined_acronyms.items():
            text = re.sub(fr'\b{acronym}\b', expansion, text)
        return text

    @staticmethod
    def _snap_to_word_boundaries(text, entities):
        """Extend every span outwards to whole words.

        Transformer spans can stop inside a word. dslim/bert-base-NER
        returns "Okaf" for "Okafor" at 0.98 confidence, in every sentence
        tried -- the tokenizer splits it into subwords and the aggregation
        does not reach the end. Replacing that span leaves "or" sitting in
        the output next to the pseudonym: the name is redacted except for
        its tail, which on a longer or rarer name is a good deal more
        identifying than two letters.

        Fixed here rather than by changing aggregation_strategy because it
        holds for any model and any strategy, and because a span that
        already ends on a boundary is left exactly as it was: the loop only
        moves while the neighbouring character is part of the same word.
        """
        snapped = []
        for ent in entities:
            start, end = ent["start"], ent["end"]
            while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
                start -= 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            if (start, end) != (ent["start"], ent["end"]):
                ent = {**ent, "start": start, "end": end, "text": text[start:end]}
            snapped.append(ent)
        return snapped

    # Words that can start a capitalised run without being part of a name.
    _NOT_NAME_WORDS = frozenset("""a an the at on in of for to from by with and or but as is was were
        mr mrs ms dr prof sir madam our their his her its this that these those my your we they he she it
        i on monday tuesday wednesday thursday friday saturday sunday january february march april may
        june july august september october november december""".split())
    _NAME_WORD = re.compile(r"[A-Z][\w&'.-]*")

    @classmethod
    def _extend_names(cls, text, entities):
        """Widen person and organisation spans over directly adjacent
        capitalised words the model left out.

        The general model returns "Andrade S/A" for "Cassiano Andrade S/A"
        often enough to matter: the rest of the name then stays in the
        output next to its replacement. A run of capitalised words is one
        name far more often than two, so a PER/ORG span takes in capitalised
        neighbours separated by a single space -- stopping at punctuation,
        lower-case words, function words, a sentence's first word and other
        entities.
        """
        spans = sorted(entities, key=lambda e: e["start"])
        for i, ent in enumerate(spans):
            if ent.get("entity_group") not in ("PER", "ORG"):
                continue
            left_limit = spans[i - 1]["end"] if i > 0 else 0
            right_limit = spans[i + 1]["start"] if i + 1 < len(spans) else len(text)
            start, end = ent["start"], ent["end"]
            while True:
                m = re.search(r"([A-Z][\w&'.-]*) $", text[left_limit:start])
                if not m or m.group(1).rstrip(".").lower() in cls._NOT_NAME_WORDS:
                    break
                word_start = left_limit + m.start(1)
                # A sentence's first word is capitalised whatever it is
                # ("Contact Jonas Berg"), so it is not evidence of a name.
                before = text[:word_start].rstrip(" ")
                if not before or before[-1] in ".!?:;\n\"'(":
                    break
                start = word_start
            while True:
                m = re.match(r" ([A-Z][\w&'.-]*)", text[end:right_limit])
                if not m or m.group(1).rstrip(".").lower() in cls._NOT_NAME_WORDS:
                    break
                end += m.end(1)
            if (start, end) != (ent["start"], ent["end"]):
                spans[i] = {**ent, "start": start, "end": end, "text": text[start:end]}
        return spans

    def clean_entities(self, text, ner_entities, regex_entities):
        ner_entities = self._snap_to_word_boundaries(text, ner_entities)
        regex_entities = self._snap_to_word_boundaries(text, regex_entities)

        # Merge all sources: language model + cyber model + regex.
        for ent in ner_entities:
            ent.setdefault('_source', 'model')
        for ent in regex_entities:
            # Custom types arrive already tagged (see custom_entities.detect);
            # everything else in this list came from SENSITIVE_PATTERNS.
            ent.setdefault('_source', 'regex')
        all_entities = ner_entities + regex_entities

        # Sort by start position; longest match first on ties; and on the
        # same span, by how exact the source is.
        #
        # That last tie-break matters more than it looks. The general NER
        # model labels "LockBit" ORG, while SENSITIVE_PATTERNS names it
        # MALWARE explicitly. Both matched the identical span, so which one
        # survived was decided by list order -- and the model's label meant
        # a malware family was redacted as though it were a company.
        #
        # The order is regex, then custom type, then model:
        #
        #   regex   fires only on an exact format -- a CVE id, a hex hash of
        #           a known length, a name from a literal list. When it
        #           matches it cannot be wrong about what it matched.
        #   custom  a type the owner named themselves, matched by meaning.
        #           A judgement, but their own explicit statement about
        #           their own data, so it beats a general-purpose guess.
        #   model   trained on someone else's corpus and applied to this
        #           one. Right most of the time, and the thing to give way
        #           when something more specific disagrees.
        _source_rank = {'regex': 0, 'custom': 1, 'model': 2}
        all_entities.sort(key=lambda x: (
            x['start'],
            -(x['end'] - x['start']),
            _source_rank.get(x.get('_source'), 1),
        ))

        kept = []
        last_end = -1
        for ent in all_entities:
            # Only keep entities that don't overlap a previously kept one.
            if ent['start'] >= last_end:
                kept.append(ent)
                last_end = ent['end']

        # NER pipelines often split one real-world name at punctuation --
        # e.g. "Fatima Al-Sayed" or "CERT-EU" come back as two touching
        # same-label entities instead of one. Left unmerged, "mask" mode
        # emits a separate "xxxx" per fragment with no separator between
        # them (garbled output like "xxxxxxxx"). Merge same-label entities
        # that are adjacent or joined only by a single space/hyphen.
        kept = self._extend_names(text, kept)

        final_entities = []
        for ent in kept:
            if final_entities:
                prev = final_entities[-1]
                gap = text[prev['end']:ent['start']]
                if ent.get('entity_group') == prev.get('entity_group') and gap in ("", " ", "-"):
                    prev['end'] = ent['end']
                    prev['text'] = text[prev['start']:prev['end']]
                    prev['score'] = min(prev.get('score', 1.0), ent.get('score', 1.0))
                    continue
            # Assign the friendly display type (e.g. "PER" -> "Person").
            ent['readable_type'] = diccionarios.label_map.get(ent['entity_group'], ent['entity_group'])
            # Flagged rather than dropped: the caller still gets told a CVE
            # or a malware family was found, it simply is not redacted.
            ent['preserved'] = ent['readable_type'] in diccionarios.preserved_types()
            ent.pop('_source', None)   # internal to the merge above
            final_entities.append(ent)

        return final_entities

    def replace_entities(self, text, entities, mode="hash", hash_key=None, scope="adhoc"):
        """``hash_key`` is required for mode="hash" and mode="pseudonymise",
        and ignored otherwise. ``scope`` (the dataset id) fixes the date
        offset and the surrogates for mode="pseudonymise" -- see pseudonymise.py.

        It is a mandatory argument rather than an optional one with a
        default because the previous implementation derived pseudonyms from
        an unkeyed MD5, which anyone holding the output could reverse by
        brute force -- see ``pseudonym`` for the full reasoning. Raising
        here is what prevents a caller from silently reintroducing that.
        """
        if not entities: return text
        if mode == "pseudonymise" and not hash_key:
            raise ValueError("mode='pseudonymise' requires a hashing key; see pseudonymise.py.")
        parts = []
        last_end = 0
        for ent in sorted(entities, key=lambda x: x['start']):
            start, end = ent['start'], ent['end']
            parts.append(text[last_end:start])
            if mode == "pseudonymise":
                replacement = pseudonymise.surrogate(
                    ent['text'], ent.get('readable_type') or ent.get('entity_group'), hash_key, scope,
                )
            elif mode == "hash":
                replacement = pseudonym.hash_token(
                    ent['text'], ent.get('readable_type', 'ENTITY'), hash_key,
                )
            elif mode == "label":
                tag = ent.get('readable_type') or ent.get('entity_group') or 'ENTITY'
                replacement = f"[{tag.upper()}]"
            elif mode == "mask":
                replacement = "xxxx"
            else:
                replacement = ent['text']
            parts.append(replacement)
            last_end = end
        parts.append(text[last_end:])
        return "".join(parts)

    def replace_with_fake_data(self, text, entities):
        """Substitution: every entity replaced by a realistic stand-in.

        The stand-ins come from ``pseudonymise.surrogate``, the generators
        per-type pseudonymisation uses, under a key made for this call and
        then discarded. The same value therefore gets the same stand-in
        throughout one text, and nothing links two calls or leads back.

        Substitution used to keep its own table of generators. It covered
        only the original types: every type added since (ages, usernames,
        phone numbers, addresses...) fell through to ``fake.word()``, so
        "45-year-old" could come out as "must" -- a random English word
        that reads as prose, with nothing to show it is junk. ``surrogate``
        answers an unknown type with a ``[TYPE]`` label instead, and keeps
        the rules learned there: documentation IP ranges, ``example.com``
        domains, digits replaced but the shape kept.
        """
        if not entities: return text
        key = pseudonym.ephemeral_key()
        parts = []
        last_end = 0
        for ent in sorted(entities, key=lambda x: x['start']):
            parts.append(text[last_end:ent['start']])
            parts.append(pseudonymise.surrogate(
                ent['text'], ent.get('readable_type') or ent.get('entity_group'), key, scope="substitution",
            ))
            last_end = ent['end']
        parts.append(text[last_end:])
        return "".join(parts)