"""The "pseudonymisation" technique: a strategy per entity type.

The other techniques treat every entity alike -- a label, a mask, a fake
value or a hash. That removes the identifier and much of what made the
report useful: a timeline reduced to "[DATE] ... two days later, on [DATE]"
cannot be followed, and substitution gave the same person a different fake
name at every mention.

Here each type gets the treatment the S-CECILIA annotation guidelines give
it:

  replace      consistent, keyed surrogates: the same value becomes the same
               fake value throughout a dataset (names, organisations,
               places, emails, usernames, addresses, hosts, files, IPs);
               digit strings keep their shape (phones, ID and card numbers)
  shift        dates move by a secret number of days and clock times by a
               secret number of minutes, the same across the dataset, so
               the days between dates (and the time between times) are kept
  generalise   ages become decade ranges, job titles broad roles, amounts
               of money are rounded
  suppress     health information, nationality, credentials

Everything is keyed. The surrogate for a value, and the date offset, are
derived with HMAC-SHA256 from the tenant's key in Vault (see pseudonym.py)
and a scope -- the dataset id. Without the key the offset cannot be worked
out from the output: knowing one real date and its shifted form reveals the
offset for that one dataset only. With the key the output is reproducible,
which is what makes it pseudonymisation in the GDPR sense (Art. 4(5)) rather
than anonymisation: the output is still personal data for anyone who holds,
or can obtain, the key.

A value this module cannot parse is never left in clear text: it falls back
to the type label.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import hmac
import re

from faker import Faker

# Offsets: at least a month (a shift of days hides little), at most five
# years, either direction, never zero.
MIN_SHIFT_DAYS, MAX_SHIFT_DAYS = 30, 5 * 365
MIN_SHIFT_MINUTES, MAX_SHIFT_MINUTES = 17, 11 * 60 + 43

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_MONTH_INDEX = {m.lower(): i + 1 for i, m in enumerate(_MONTHS)}
_MONTH_INDEX.update({m[:3].lower(): i + 1 for i, m in enumerate(_MONTHS)})
_MONTH_INDEX["sept"] = 9
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_RE = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
_ORD = r"(?:st|nd|rd|th)?"


def _hmac_int(key: str, message: str) -> int:
    return int.from_bytes(hmac.new(bytes.fromhex(key), message.encode("utf-8"), hashlib.sha256).digest()[:8], "big")


def _signed_offset(key: str, message: str, low: int, high: int) -> int:
    n = _hmac_int(key, message)
    magnitude = low + n % (high - low + 1)
    return magnitude if (n >> 32) & 1 else -magnitude


def date_offset_days(key: str, scope: str) -> int:
    """The secret, per-scope shift applied to every date in that scope."""
    return _signed_offset(key, f"date-shift:{scope}", MIN_SHIFT_DAYS, MAX_SHIFT_DAYS)


def time_offset_minutes(key: str, scope: str) -> int:
    """The shift for clock times that come without a date."""
    return _signed_offset(key, f"time-shift:{scope}", MIN_SHIFT_MINUTES, MAX_SHIFT_MINUTES)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def _match_case(template: str, word: str) -> str:
    if template.isupper():
        return word.upper()
    if template[:1].islower():
        return word.lower()
    return word


def _month_name(template: str, month: int) -> str:
    full = _MONTHS[month - 1]
    t = template.rstrip(".")
    name = full[:len(t)] if len(t) <= 4 and t.lower() != full.lower() else full
    if t.lower() == "sept":
        name = "Sept" if month == 9 else full[:3]
    return _match_case(t, name) + ("." if template.endswith(".") else "")


def _ordinal(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _shift_iso_datetime(m, days, minutes_unused):
    date = dt.date(int(m["y"]), int(m["m"]), int(m["d"])) + dt.timedelta(days=days)
    return f"{date.year:04d}{m['s1']}{date.month:02d}{m['s2']}{date.day:02d}{m['rest'] or ''}"


def shift_date(value: str, days: int) -> str | None:
    """``value`` moved by ``days``, in the same format, or None if unparseable."""
    # Entity text often carries the sentence's punctuation ("29 February.");
    # parse without it and leave it where it was.
    v = value.strip().rstrip(".,;:!?)]")
    try:
        # 2024-02-27, 2024/02/27, 2024-02-27T14:32:07Z (the time part is kept:
        # the whole datetime moves by whole days, so its time of day is right)
        m = re.fullmatch(r"(?P<y>\d{4})(?P<s1>[-/.])(?P<m>\d{1,2})(?P<s2>[-/.])(?P<d>\d{1,2})(?P<rest>[T ].*)?", v)
        # With a clock time attached it is a datetime: shift_datetime moves
        # the time too, so it agrees with the same time written on its own.
        if m and not (m["rest"] and re.search(r"\d", m["rest"])):
            return value.replace(v, _shift_iso_datetime(m, days, 0))
        # 20240227
        m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", v)
        if m and 1900 <= int(m[1]) <= 2100:
            d = dt.date(int(m[1]), int(m[2]), int(m[3])) + dt.timedelta(days=days)
            return value.replace(v, f"{d.year:04d}{d.month:02d}{d.day:02d}")
        # 27/02/2024, 02-27-24, 27.02.2024 -- day first unless that is impossible
        m = re.fullmatch(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{2}|\d{4})", v)
        if m:
            a, b, y = int(m[1]), int(m[3]), m[4]
            year = int(y) + (2000 if len(y) == 2 and int(y) < 70 else 1900 if len(y) == 2 else 0)
            day_first = not (b > 12 and a <= 12)
            day, month = (a, b) if day_first else (b, a)
            d = dt.date(year, month, day) + dt.timedelta(days=days)
            yy = f"{d.year % 100:02d}" if len(y) == 2 else f"{d.year:04d}"
            first, second = (d.day, d.month) if day_first else (d.month, d.day)
            w1, w2 = len(m[1]), len(m[3])
            return value.replace(v, f"{first:0{w1}d}{m[2]}{second:0{w2}d}{m[2]}{yy}")
        # February 27th, 2024 / Feb 27 2024 / February 27
        m = re.fullmatch(rf"(?P<mon>{_MONTH_RE})\s+(?P<d>\d{{1,2}})(?P<ord>{_ORD})(?P<sep>,?\s+)?(?P<y>\d{{4}})?", v, re.IGNORECASE)
        if m:
            year = int(m["y"]) if m["y"] else 2000        # a leap year, so 29 February exists
            d = dt.date(year, _MONTH_INDEX[m["mon"].rstrip(".").lower()], int(m["d"])) + dt.timedelta(days=days)
            out = f"{_month_name(m['mon'], d.month)} {d.day}{_ordinal(d.day) if m['ord'] else ''}"
            if m["y"]:
                out += f"{m['sep'] or ' '}{d.year}"
            return value.replace(v, out)
        # 27 February 2024 / 27th Feb 2024 / 27 February
        m = re.fullmatch(rf"(?P<d>\d{{1,2}})(?P<ord>{_ORD})\s+(?:of\s+)?(?P<mon>{_MONTH_RE})(?P<sep>,?\s+)?(?P<y>\d{{4}})?", v, re.IGNORECASE)
        if m:
            year = int(m["y"]) if m["y"] else 2000
            d = dt.date(year, _MONTH_INDEX[m["mon"].rstrip(".").lower()], int(m["d"])) + dt.timedelta(days=days)
            out = f"{d.day}{_ordinal(d.day) if m['ord'] else ''} {_month_name(m['mon'], d.month)}"
            if m["y"]:
                out += f"{m['sep'] or ' '}{d.year}"
            return value.replace(v, out)
        # March 2023
        m = re.fullmatch(rf"(?P<mon>{_MONTH_RE})(?P<sep>,?\s+)(?P<y>\d{{4}})", v, re.IGNORECASE)
        if m:
            d = dt.date(int(m["y"]), _MONTH_INDEX[m["mon"].rstrip(".").lower()], 15) + dt.timedelta(days=days)
            return value.replace(v, f"{_month_name(m['mon'], d.month)}{m['sep']}{d.year}")
        # March
        m = re.fullmatch(_MONTH_RE, v, re.IGNORECASE)
        if m:
            d = dt.date(2001, _MONTH_INDEX[v.rstrip(".").lower()], 15) + dt.timedelta(days=days)
            return value.replace(v, _month_name(v, d.month))
        # Monday
        if v.lower() in (w.lower() for w in _WEEKDAYS):
            i = [w.lower() for w in _WEEKDAYS].index(v.lower())
            return value.replace(v, _match_case(v, _WEEKDAYS[(i + days) % 7]))
        # 2023 (a bare year)
        m = re.fullmatch(r"(19|20)\d{2}", v)
        if m:
            return value.replace(v, str(int(v) + round(days / 365.25)))
    except ValueError:        # 31/02/2024 and the like
        return None
    return None


# ---------------------------------------------------------------------------
# Clock times
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"(?P<h>\d{1,2})(?P<sep>[:.])(?P<m>\d{2})(?:(?P<sep2>[:.])(?P<s>\d{2}))?(?P<frac>\.\d+)?"
    r"(?P<sp>\s*)(?P<ampm>[AaPp]\.?[Mm]\.?)?")


def shift_time(value: str, minutes: int) -> str | None:
    """Every clock time in ``value`` moved by ``minutes``; None if there is none.

    Time-zone words ("UTC", "EST") and the rest of the text are kept.
    """
    def one(m):
        h, mi = int(m["h"]), int(m["m"])
        ampm = m["ampm"]
        if mi > 59 or (ampm and not 1 <= h <= 12) or (not ampm and h > 23):
            raise ValueError
        if ampm:
            h = h % 12 + (12 if ampm[0].lower() == "p" else 0)
        total = (h * 60 + mi + minutes) % (24 * 60)
        nh, nm = divmod(total, 60)
        if ampm:
            shown = nh % 12 or 12
            suffix = ("PM" if nh >= 12 else "AM")
            suffix = _match_case(ampm, suffix) if "." not in ampm else ("p.m." if nh >= 12 else "a.m.")
            hour = f"{shown:02d}" if m["h"].startswith("0") else str(shown)
        else:
            suffix, hour = "", f"{nh:0{len(m['h'])}d}"
        out = f"{hour}{m['sep']}{nm:02d}"
        if m["s"]:
            out += f"{m['sep2']}{m['s']}"
        return out + (m["frac"] or "") + (m["sp"] + suffix if ampm else m["sp"])

    if not _TIME_RE.search(value):
        return None
    try:
        return _TIME_RE.sub(one, value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Consistent surrogates
# ---------------------------------------------------------------------------

def _faker(key: str, readable_type: str, value: str) -> Faker:
    f = Faker()
    f.seed_instance(_hmac_int(key, f"surrogate:{readable_type}:{value.strip().lower()}"))
    return f


def _same_shape_digits(key: str, readable_type: str, value: str) -> str:
    """Every digit replaced, deterministically; separators and letters kept."""
    stream = hmac.new(bytes.fromhex(key), f"digits:{readable_type}:{value}".encode(), hashlib.sha256).digest()
    out, k = [], 0
    for ch in value:
        if ch.isdigit():
            out.append(str(stream[k % len(stream)] % 10)); k += 1
        else:
            out.append(ch)
    return "".join(out)


def _documentation_ip(key: str, value: str) -> str:
    n = _hmac_int(key, f"ip:{value}")
    if ":" in value and value.count(":") > 1:           # IPv6 -> 2001:db8::/32
        return "2001:db8:%x:%x::%x" % (n & 0xffff, (n >> 16) & 0xffff, ((n >> 32) & 0xfffe) + 1)
    block = ("192.0.2", "198.51.100", "203.0.113")[n % 3]   # RFC 5737
    return f"{block}.{(n >> 8) % 254 + 1}"


# First match wins: executive words, then the specialism (so a "lead system
# administrator" is IT staff, not a manager), then generic management.
_ROLE_GROUPS = [
    (re.compile(r"\b(chief|ceo|cfo|cto|ciso|cio|coo|president|director|head|vice|vp|executive|founder)\b", re.I), "senior manager"),
    (re.compile(r"\b(security|soc|cyber|forensic|incident|threat|penetration|pentester)\b", re.I), "security specialist"),
    (re.compile(r"\b(admin\w*|engineer|developer|programmer|technician|architect|devops|sysadmin|it|network|database|dba|support)\b", re.I), "IT staff member"),
    (re.compile(r"\b(manager|supervisor|lead|coordinator|officer)\b", re.I), "manager"),
    (re.compile(r"\b(consultant|contractor|advisor|adviser|auditor)\b", re.I), "consultant"),
    (re.compile(r"\b(doctor|nurse|physician|surgeon|patient|teacher|professor|student|lawyer|journalist|accountant)\b", re.I), "professional"),
]


def generalise_role(value: str) -> str:
    for pattern, group in _ROLE_GROUPS:
        if pattern.search(value):
            return group
    return "staff member"


def generalise_age(value: str) -> str | None:
    m = re.search(r"\d{1,3}", value)
    if not m or not 0 < int(m.group()) < 120:
        return None
    low = int(m.group()) // 10 * 10
    return value[:m.start()] + f"{low}–{low + 9}" + value[m.end():]


_MONEY_RE = re.compile(r"(?P<num>\d[\d,]*(?:\.\d+)?)(?P<rest>\s*(?:thousand|million|billion|[kKmMbB]n?)?\b)?")


def round_money(value: str) -> str | None:
    m = _MONEY_RE.search(value)
    if not m:
        return None
    number = float(m["num"].replace(",", ""))
    if number <= 0:
        return None
    digits = len(str(int(number))) - 1
    rounded = round(number, -digits) if number >= 10 else round(number)
    shown = f"{int(rounded):,}" if "," in m["num"] or rounded >= 10000 else f"{int(rounded)}"
    return "about " + value[:m.start("num")] + shown + value[m.end("num"):]


SUPPRESS = {
    "MEDICAL_CONDITION": "[HEALTH INFORMATION]",
    "NATIONALITY": "[NATIONALITY]",
    "PASSWORD": "[CREDENTIAL]",
}


def shift_datetime(value: str, days: int, minutes: int) -> str | None:
    """A date, a time, or both ("27 February 2024 at 14:32",
    "2024-02-27T14:32:07Z"), moved consistently.

    Every date moves by exactly ``days`` -- with or without a time attached
    -- so the number of days between any two dates is kept. Every clock time
    moves by ``minutes`` around the clock, so a time here and the same time
    written alone elsewhere shift to the same value, and differences between
    times are kept except across midnight. The date is deliberately not
    rolled over when a shifted time crosses midnight: that would make a
    datetime and a plain date on the same day land on different days, and
    "two days later" would read as one.
    """
    shifted = shift_date(value, days)
    if shifted is not None:
        return shifted
    tm = _TIME_RE.search(value)
    if not tm:
        return None
    head = re.match(r"^(?P<core>.*?)(?P<conn>\s*(?:\bat\b|,|T)?\s*)$", value[:tm.start()], re.DOTALL)
    core, conn = head["core"], head["conn"]
    if not core.strip():
        return shift_time(value, minutes)
    new_date = shift_date(core, days)
    new_time = shift_time(value[tm.start():], minutes)
    if new_date is None or new_time is None:
        return None
    return new_date + conn + new_time


def surrogate(value: str, readable_type: str, key: str, scope: str) -> str:
    """What ``value`` of ``readable_type`` becomes. Never ``value`` itself."""
    t = (readable_type or "").upper()
    label = f"[{t or 'ENTITY'}]"
    if t in SUPPRESS:
        return SUPPRESS[t]
    if t in ("DATE", "TIME"):
        # Dates, times and datetimes all go through one function, so every
        # temporal value in a dataset moves by the same secret amount.
        shifted = shift_datetime(value, date_offset_days(key, scope), time_offset_minutes(key, scope))
        return shifted if shifted and shifted != value else f"[{t}]"
    if t == "AGE":
        return generalise_age(value) or label
    if t == "JOB_TITLE":
        return generalise_role(value)
    if t in ("AMOUNT_OF_MONEY", "QUANTITY"):
        return round_money(value) or label
    if t in ("PHONE_NUMBER", "IDENTIFICATION_NUMBER", "BANK_OR_CARD_NUMBER"):
        return _same_shape_digits(key, t, value) if re.search(r"\d", value) else label
    if t == "IP":
        return _documentation_ip(key, value)
    f = _faker(key, t, value)
    if t == "PER":
        return f.name()
    if t == "ORG":
        return f.company()
    if t == "LOC":
        return f.city()
    if t == "STREET_ADDRESS":
        return f.street_address()
    if t == "EMAIL":
        return f"{f.user_name()}@example.com"
    if t == "USERNAME":
        return f"{f.user_name()}{f.random_int(10, 99)}"
    if t in ("DOMAIN", "HOST"):
        return f"{f.domain_word()}.example.com"
    if t == "URL":
        return f"https://www.example.com/{f.uri_path()}"
    if t == "FILE":
        ext = value.rsplit(".", 1)[-1] if "." in value else "bin"
        return f"{f.word()}.{ext}"
    if t == "FILE_PATH":
        return f"/redacted/{f.word()}/{f.word()}"
    return label
