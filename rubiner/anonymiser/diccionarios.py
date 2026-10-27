import os
import re

# --- 1. PREDEFINED ACRONYMS (CLEANED UP) ---
# SCADA, PII, NSA, VPN, IoT, etc. were removed to avoid expansions that confuse the model.
# Only obscure organisations or acronyms that NEED contextual explanation are kept.
predefined_acronyms = {
    'FONASA': 'Chilean National Health Fund', 'BlackCat': 'Notorious ransomware group',
    'BfV': 'German Federal Office for the Protection of the Constitution',
    'RIA': 'Estonian Information System Authority', 'SBB': 'Swiss Federal Railways',
    'SPD': 'Social Democratic Party of Germany',
    'DVP': 'Free Democratic Party (FDP) parliamentary group',
    'GRU': 'Main Intelligence Directorate of the Ministry of Defense of the Russian Federation',
    'CERT-UA': 'Ukrainian Computer Emergency Response Team',
    'NSC': 'National Security Committee of the Republic of Kazakhstan',
    'FSB': 'Federal Security Service (Russia)',
    'IRGC-IO': 'Intelligence Organization of the Islamic Revolutionary Guard Corps',
    'TGR': 'General Treasury of the Republic of Chile',
    'STISC': 'Moldovan Information Technology and Cybersecurity Service',
    'ACTINIUM': 'Russian state-sponsored hacking group (aka Gamaredon)', 'NSM': 'Norwegian National Security Authority',
    'CNI': "Spain's National Intelligence Centre", 'UMB': 'Matej - Bel University',
    'SESARAM': 'Regional Health Service of the Autonomous Region of Madeira', 'IRS': 'Internal Revenue Service',
    'CGPJ': 'General Council of the Judiciary',
    'AEAT': 'Spanish Tax Administration Agency',
    'GURMO': 'Main Intelligence Directorate of the Ministry of Defense of Ukraine',
    'PJSC': 'Public Joint Stock Company',
    'SOFCOM': 'Systems integrator company working with Russian government and corporate clients',
    'JSC': 'Joint Stock Company', 'BIFIT': 'Financial services company enabling remote banking',
    'NOBELIUM': 'Also known as APT29, Russian state-sponsored hacking group',
    'SVK': 'Polish Military Counterintelligence Service',
    'CERT Polska': 'Computer Emergency Response Team Poland', 'CBZC': 'Polish Central Bureau for Combating Cybercrime',
    'UAC-0114': 'Tracking name for Winter Vivern by Ukraine\'s CERT', 'SBU': 'Ukrainian Security Service',
    'AGID': 'Agency for Digital Italy',
    'CCSC': 'Capital Health Communication Center',
    'YMPH': 'Yamaha Motor Philippines', 'dena': 'German Energy Agency', 'GTsST': 'Main Center for Special Technologies',
    'NSPK': 'Russian National Payment Card System',
    'IRoX': 'Hacktivist group',
    'NUKIB': 'National Cyber and Information Security Agency of Czechia',
    'NISC': 'National Center of Incident Readiness and Strategy for Cybersecurity',
    'MosgorBTI': 'Moscow Register of Deeds',
    'DSS': 'Norwegian Government Security and Service Organisation',
    'NCSC-NO': 'Norwegian National Cyber Security Centre',
    'ASP': 'Italian Provincial Health Authority of Cosenza', 'APOC': 'Administrative Office of Pennsylvania Courts',
    'USCYBERCOM': 'US Cyber Command',
    'ZAC NRW': 'North Rhine-Westphalia Cybercrime Centre',
    'BSI': 'German Federal Office for Information Security',
    'ANSSI': 'French cybersecurity agency', 'EPS': 'Colombian health insurance provider',
    'FSO': 'Federal Protective Service',
    'GUR': "Ukraine's Defence Intelligence Directorate", 'SVR': 'Russia\'s Foreign Intelligence Service',
    'IZIS': 'Integrated Health Information System',
    'FFAA': 'General Directorate of Information and Communication Technologies of the Paraguayan Armed Forces'
}

# --- 2. LABEL MAPPING ---
# Unifies the outputs of different models (HuggingFace, Spacy, etc.) into a standard label set.
label_map = {
    "PERSON": "PER", "PER": "PER", "PER_FIX": "PER",
    "LOCATION": "LOC", "GPE": "LOC", "LOC": "LOC", "LOC_FIX": "LOC",
    "ORGANIZATION": "ORG", "ORG": "ORG", "GOV_AUTHORITIES": "ORG",
    # TIME is its own type: it has its own entry in ENTITY_CATALOGUE, and
    # folding it into DATE made the "Times of day" setting do nothing --
    # times were kept or redacted by the Dates checkbox instead.
    "DATE": "DATE", "TIME": "TIME",
    "EMAIL": "EMAIL",
    "IP": "IP", "IP_ADDRESS": "IP", "IPv6": "IP",
    "HASH": "HASH", "CRYPTO_HASH": "HASH",
    "HOSTNAME": "HOST",
    "DOMAIN": "DOMAIN",
    "CVE": "CVE", "CVE_ID": "CVE",
    "MALWARE": "MALWARE", "CYBER_GROUP": "MALWARE",
    "EXECUTABLE": "FILE", "FILE": "FILE", "FILE_PATH": "FILE_PATH",
    "TECH_ACRONYM": "TECH", "PROTECTED_TERMS": "TECH",
    "MISC": "MISC", "CRYPTO_CURR": "TECH"
}

# --- 3. DATE PATTERNS ---
# Covers numeric formats, English month names, and date ranges.
DATE_PATTERNS = [

    # Months
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
               re.IGNORECASE),

    # Days of the week
    re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE),


    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),
    re.compile(r"\b\d{8}\b"),
    # English dates
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
               re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b", re.IGNORECASE),
    # English date ranges
    re.compile(r"\b(?:from|between)\s+(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
               r"(?:\s+and|to|-)\s+(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
               r"(?:\s+\d{2,4})?\b", re.IGNORECASE),
    # ISO timestamps
    re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
]

# --- 4. SENSITIVE PATTERNS (CYBERSECURITY & PRIVACY) ---
# ---------------------------------------------------------------------------
# What is sensitive, and what is the point of the report
# ---------------------------------------------------------------------------
#
# Detection and redaction are separate questions. SENSITIVE_PATTERNS below
# matches CVE ids, malware family names and file hashes, and it should --
# knowing they are there is useful, and the entity list reports them. But
# they are public identifiers, not personal data, and for an incident report
# they are the reason another CERT reads it at all. Redacting them produces
# output like:
#
#   "On <hash> the <hash> team coordinated with <hash> after <hash> was
#    exploited against <hash> clients."
#
# which has had the intelligence removed and the grammar left intact. The
# people are protected either way; the difference is whether the report is
# still worth sending.
#
# Types listed here are detected, reported in the entity list with
# ``preserved: True``, and left untouched in the text.
#
# Deliberately NOT preserved, despite being arguable: IP addresses and
# domains. In threat intelligence an attacker's infrastructure is exactly
# what you want to share, but a victim's is exactly what you must not, and
# nothing in the text distinguishes them. Redacting both is the safe
# default; a tenant that shares only attacker infrastructure can add them
# via the environment variable.
#
# Override with ANONYMISER_PRESERVE_TYPES, a comma-separated list of types
# as they appear in ``label_map``'s values. Set it to an empty string to
# preserve nothing and restore the previous behaviour.
# Every entity type the pipeline can produce, in the order a person would
# want to read them, with a label and a line of help. This is the catalogue
# the settings screen renders; anything detectable but absent here would be
# invisible to the person choosing, so a new type in ``label_map`` belongs
# here too (``test_every_entity_type_is_configurable`` checks that).
ENTITY_CATALOGUE = [
    ("PER",       "People",            "Names of individuals."),
    ("ORG",       "Organisations",     "Companies, teams, institutions."),
    ("LOC",       "Locations",         "Places, cities, countries."),
    ("EMAIL",     "Email addresses",   ""),
    ("IP",        "IP addresses",      "Both IPv4 and IPv6."),
    ("DOMAIN",    "Domains",           "example.com and similar."),
    ("URL",       "URLs",              ""),
    ("HOST",      "Hostnames",         ""),
    ("DATE",      "Dates and times",   "Redacting these can make an incident timeline unreadable."),
    ("FILE",      "File names",        "Executables and similar."),
    ("FILE_PATH", "File paths",        "Paths can contain a user's name."),
    ("CVE",       "CVE identifiers",   "Public vulnerability ids. Usually the point of the report."),
    ("MALWARE",   "Malware and actors","Family and threat-actor names. Public, and usually the point."),
    ("HASH",      "File hashes",       "Indicators of compromise, meant to be shared."),
    ("TECH",      "Technology terms",  "Products, protocols, tools."),
    ("MISC",      "Other",             "Anything detected that fits no other type."),
    # The incident-report pack (custom_entities.INCIDENT_REPORT_PACK): personal
    # data incident reports carry that the transformer models have no label
    # for. Kept in the same order as the pack; test_pack_types_are_catalogued
    # checks the two agree.
    ("JOB_TITLE",             "Job titles",              "A person's role, e.g. system administrator."),
    ("AGE",                   "Ages",                    "Including '45-year-old' and 'age 42'."),
    ("USERNAME",              "Usernames",               "Account names and login handles."),
    ("PASSWORD",              "Passwords and credentials", "Passwords, PINs, tokens."),
    ("IDENTIFICATION_NUMBER", "Identification numbers",  "Passport, licence, national and employee IDs."),
    ("BANK_OR_CARD_NUMBER",   "Bank and card details",   "Card numbers, expiry dates, CVVs, account numbers."),
    ("PHONE_NUMBER",          "Phone numbers",           ""),
    ("MEDICAL_CONDITION",     "Health information",      "Conditions, diagnoses, treatments."),
    ("NATIONALITY",           "Nationality and ethnicity", ""),
    ("TIME",                  "Times of day",            "Clock times and timestamps."),
    ("SOFTWARE",              "Software",                "Products and versions. Technical content, usually kept."),
    ("OPERATING_SYSTEM",      "Operating systems",       "Technical content, usually kept."),
    ("STREET_ADDRESS",        "Street addresses",        "Streets, house numbers, postcodes."),
    ("AMOUNT_OF_MONEY",       "Amounts of money",        ""),
    ("QUANTITY",              "Quantities",              "Counts of people, devices or records; data sizes."),
]

ALL_ENTITY_TYPES = frozenset(t for t, _, _ in ENTITY_CATALOGUE)

# The catalogue before the incident-report pack was added. A redaction
# selection saved then knew only these, so a later type missing from it is
# not a decision to keep it (see dataset_store.get_redaction_types).
LEGACY_ENTITY_TYPES = frozenset({
    "PER", "ORG", "LOC", "EMAIL", "IP", "DOMAIN", "URL", "HOST",
    "DATE", "FILE", "FILE_PATH", "CVE", "MALWARE", "HASH", "TECH", "MISC",
})


# Technology terms (products, protocols, tools), software and operating
# systems identify no one and are the technical content of a report, like
# CVE ids -- kept by default. Redacting TECH turned "the VPN concentrator"
# into "the [TECH] concentrator" and "an email" into "an [TECH]".
DEFAULT_PRESERVED_TYPES = frozenset({"CVE", "MALWARE", "HASH", "TECH", "SOFTWARE", "OPERATING_SYSTEM"})


def preserved_types() -> set:
    raw = os.getenv("ANONYMISER_PRESERVE_TYPES")
    if raw is None:
        return set(DEFAULT_PRESERVED_TYPES)
    return {t.strip().upper() for t in raw.split(",") if t.strip()}


SENSITIVE_PATTERNS = {
    "EMAIL": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
    "URL": re.compile(r"\bhttps?://?(www\.)?[\w\-]+\.[a-z]{2,}(/\S+)?\b", re.IGNORECASE),
    "IP_ADDRESS": re.compile(
        r'\b(?:(?:\d{1,3}\.){3}\d{1,3}|(?:[0-9a-fA-F]{1,4}:){2,}(?:[0-9a-fA-F]{1,4}|:)|fe80::[0-9a-fA-F:]+)\b',
        re.IGNORECASE
    ),
    "EXECUTABLE": re.compile(r'\b[\w\-\.]+\.(?:exe|dll|sh|py|bat|ps1|bin|so|zip|rar|7z)\b', re.IGNORECASE),
    "DOMAIN": re.compile(r"\b([a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?\.)+(com|net|org|es|io|ai|cloud|gov|ru)\b",
                         re.IGNORECASE),
    "HASH": re.compile(
        r'\b[a-fA-F0-9]{64}\b|'  # SHA256 (64 chars)
        r'\b[a-fA-F0-9]{40}\b|'  # SHA1 (40 chars)
        r'\b[a-fA-F0-9]{32}\b',  # MD5 (32 chars),
        re.IGNORECASE),

    # Cybersecurity-specific
    "CVE_ID": re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE),
    "MALWARE": re.compile(
        r'\b(?:APT\d{1,3}|Emotet|TrickBot|Conti|LockBit|Clop|Cl0p|WannaCry|Petya|Lazarus|Pegasus|ShadowPad|'
        # Distinctive names only: ones that are also ordinary words (Royal,
        # Hive, Play, Medusa) would fire on unrelated text.
        r'Rhysida|ALPHV|BlackCat|BianLian|REvil|Sodinokibi|Ryuk|Black Basta|DarkSide|Qakbot|QakBot|IcedID|8Base|NotPetya)\b',
        re.IGNORECASE
    ),
    "CYBER_GROUP": re.compile(r"\b(?:Redfly|Blackfly|Grayfly|Sandworm|Fancy Bear|Nobelium|Scattered Spider|Cozy Bear)\b", re.IGNORECASE),
    # Cryptocurrencies: technical content, not organisations (the general
    # model took "Bitcoin" for one). A pattern wins over a model on the same
    # span, so these are typed TECH and kept by default.
    "CRYPTO_CURR": re.compile(r"\b(?:Bitcoin|Ethereum|Monero|Litecoin|Tether|Zcash|Dogecoin|BTC|XMR|USDT)\b"),

    # Infrastructure and files
    "FILE_PATH": re.compile(r'(?:\/var\/|\/etc\/|\/home\/|C:\\|D:\\)[\w\-\\\/\.]+', re.IGNORECASE),

    # Technical terms to protect (whitelistable)
    "TECH_ACRONYM": re.compile(
        r'\b(CSIRT|SOC|VPN|SMB(?:v[1-3])?|AWS|RDP|SSH|TCP|UDP|SIEM|MFA|AD|DNS|OT|SCADA|IPv[46]|PII)\b',
        re.IGNORECASE
    ),

}