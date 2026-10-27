"""
Keyed pseudonym generation for the "hashing" anonymisation technique.

Why this module exists
----------------------
The hashing technique used to build its pseudonyms as::

    "hash_" + hashlib.md5(f"{value.lower()}_{TYPE}").hexdigest()[:8]

with no key and no salt. That is reversible by anyone holding the
anonymised output: the input space for the entity types this app detects
(person names, organisations, email addresses, IPv4 addresses, domains,
CVE ids) is small enough to enumerate offline, so a plain dictionary or
rainbow-table attack recovers the originals without touching this service
at all. The pseudonyms were not actually protective, which matters for the
GDPR pseudonymisation claim in the WP17 use case.

They were also only 8 hex characters -- 32 bits. By the birthday bound two
distinct entities collide with ~50% probability at around 77k entities,
and ``datasets_endpoints.reveal_hash`` resolves a token by taking the
*first* mapping entry whose value matches, so a collision would silently
reveal the wrong original value.

What this does instead
----------------------
``hash_<HMAC-SHA256(key, "value_TYPE")[:16]>`` -- keyed, so the mapping
from a real value to its pseudonym cannot be computed by anyone who does
not hold the key, and 64 bits wide, so collisions are not a practical
concern.

The key lives in Vault (``vault_client.get_or_create_hash_key``), one per
tenant. Per-tenant rather than per-dataset is deliberate: a CERT analysing
several incident reports usually *wants* the same actor or indicator to
carry the same pseudonym across them, and a per-dataset key would destroy
that. The trade-off is that pseudonyms are correlatable within a tenant,
which is the intended analytic property, not a leak -- correlation across
tenants is still impossible, since the keys differ.

There is deliberately no unkeyed fallback. ``hash_token`` requires a key,
and callers that have no tenant context (the ad-hoc ``/ner/anonimizacion``
endpoint) must pass an explicitly ephemeral one via ``ephemeral_key()``.
Making the key a required argument is what stops the old unkeyed path from
quietly coming back.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

# Bytes of HMAC output rendered as hex. 8 bytes -> 16 hex chars -> 64 bits.
TOKEN_HEX_CHARS = 16

KEY_BYTES = 32  # HMAC-SHA256's block-optimal key size

TOKEN_PREFIX = "hash_"


def new_key() -> str:
    """A fresh random hashing key, hex-encoded for storage in Vault's KV
    (which holds JSON, so raw bytes need an encoding)."""
    return secrets.token_hex(KEY_BYTES)


def ephemeral_key() -> str:
    """A key that is generated, used and discarded within a single request.

    For the ad-hoc ``/ner/anonimizacion`` endpoint, which anonymises a
    pasted string for immediate display and stores nothing. Entities repeat
    consistently *within* one response (same key throughout the call) but
    the result is unlinkable across calls and can never be reversed, since
    the key is gone the moment the request ends -- which is the correct
    property for a preview tool with no dataset and no ``/reveal``.
    """
    return new_key()


def hash_token(value: str, readable_type: str, key: str) -> str:
    """The pseudonym that replaces ``value`` in anonymised output.

    ``key`` is the hex-encoded tenant key from Vault. The message keeps the
    original ``"{lowercased value}_{readable type}"`` construction so that
    the same entity text detected as two different types stays
    distinguishable, and so casing differences collapse to one pseudonym.
    """
    if not key:
        raise ValueError(
            "hash_token requires a hashing key -- there is no unkeyed fallback. "
            "Use vault_client.get_or_create_hash_key(tenant_id) for stored "
            "datasets, or pseudonym.ephemeral_key() for ad-hoc one-shot calls."
        )
    message = f"{value.lower()}_{readable_type}".encode("utf-8")
    digest = hmac.new(bytes.fromhex(key), message, hashlib.sha256).hexdigest()
    return f"{TOKEN_PREFIX}{digest[:TOKEN_HEX_CHARS]}"
