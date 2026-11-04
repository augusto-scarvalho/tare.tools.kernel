"""Portable Ed25519 receipts binding a producer to supplied artifact bytes.

A signature does not certify correctness or authorize landing. Replay admission is
the responsibility of the existing receipt owner.
"""

import base64
import binascii
import hashlib
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


SCHEMA = "tare.tools/producer-receipt/1"
DOMAIN = b"tare.tools/producer-receipt/1\0"
_SUBJECT_KEYS = {
    "work_id", "session_id", "effect_id", "binding_id",
    "fence_seq", "action_sha256", "source_sha256",
    "environment_sha256", "runtime_sha256", "profile_sha256",
    "run_id",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
}


def _canonical(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ensure_ascii=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid JSON value") from exc


def _subject(value):
    if not isinstance(value, dict) or set(value) != _SUBJECT_KEYS:
        raise ValueError("subject must have exactly the required keys")
    for key, item in value.items():
        if key == "fence_seq":
            if type(item) is not int or item < 0:
                raise ValueError("fence_seq must be a nonnegative integer")
        elif key.endswith("_sha256"):
            if not isinstance(item, str) or not _HEX.fullmatch(item):
                raise ValueError(f"{key} must be lowercase SHA-256 hex")
        elif not isinstance(item, str) or not item or len(item) > 160 or "\0" in item:
            raise ValueError(f"{key} must be a nonempty ID of at most 160 characters without NUL")
    return json.loads(_canonical(value))


def _manifest(artifacts):
    if not isinstance(artifacts, dict) or len(artifacts) > 64:
        raise ValueError("artifacts must be a mapping of at most 64 entries")
    result = {}
    seen = set()
    total = 0
    for name, data in artifacts.items():
        if not isinstance(name, str) or not name or len(name) > 240:
            raise ValueError("artifact name must be 1..240 characters")
        if not name.isascii() or any(ord(c) < 32 or ord(c) == 127 or c in '\\:<>"|?*' for c in name):
            raise ValueError(f"invalid artifact name: {name!r}")
        parts = name.split("/")
        if any(
            not part or part in (".", "..") or part.endswith((".", " "))
            or part.split(".", 1)[0].upper() in _RESERVED
            for part in parts
        ):
            raise ValueError(f"invalid artifact path: {name!r}")
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f"case-insensitive artifact name collision: {name!r}")
        seen.add(folded)
        if type(data) is not bytes:
            raise ValueError(f"artifact {name!r} must be bytes")
        total += len(data)
        if total > 8 * 1024 * 1024:
            raise ValueError("artifact bytes exceed 8 MiB")
        result[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return result


class ProducerSigner:
    def __init__(self, private_key: Ed25519PrivateKey):
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("private_key must be an Ed25519PrivateKey")
        self._private_key = private_key

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    @property
    def producer_id(self) -> str:
        return hashlib.sha256(self.public_key_bytes).hexdigest()

    def sign(self, subject: dict, artifacts: dict[str, bytes]) -> dict:
        payload = {
            "producer_id": self.producer_id,
            "subject": _subject(subject),
            "artifacts": _manifest(artifacts),
        }
        signature = self._private_key.sign(DOMAIN + _canonical(payload))
        return {
            "schema": SCHEMA, "payload": payload,
            "signature": base64.b64encode(signature).decode("ascii"),
        }


def verify_receipt(
    receipt: dict, artifacts: dict[str, bytes], expected_subject: dict,
    admitted_keys: dict[str, bytes],
) -> dict:
    if not isinstance(receipt, dict) or set(receipt) != {"schema", "payload", "signature"}:
        raise ValueError("receipt must have exactly schema, payload and signature")
    if receipt["schema"] != SCHEMA:
        raise ValueError("unsupported receipt schema")
    payload = receipt["payload"]
    if not isinstance(payload, dict) or set(payload) != {"producer_id", "subject", "artifacts"}:
        raise ValueError("payload must have exactly producer_id, subject and artifacts")
    subject = _subject(payload["subject"])
    if _canonical(subject) != _canonical(_subject(expected_subject)):
        raise ValueError("receipt subject does not match expected subject")
    producer_id = payload["producer_id"]
    if not isinstance(producer_id, str) or not _HEX.fullmatch(producer_id):
        raise ValueError("invalid producer_id")
    if not isinstance(admitted_keys, dict) or producer_id not in admitted_keys:
        raise ValueError("producer is not admitted")
    public_bytes = admitted_keys[producer_id]
    if type(public_bytes) is not bytes or len(public_bytes) != 32:
        raise ValueError("admitted public key must be 32 bytes")
    if hashlib.sha256(public_bytes).hexdigest() != producer_id:
        raise ValueError("admitted public key does not match producer_id")
    manifest = _manifest(artifacts)
    if _canonical(payload["artifacts"]) != _canonical(manifest):
        raise ValueError("artifact manifest does not match supplied bytes")
    encoded = receipt["signature"]
    if not isinstance(encoded, str):
        raise ValueError("signature must be base64")
    try:
        signature = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError("signature must be strict base64") from exc
    if len(signature) != 64:
        raise ValueError("Ed25519 signature must be 64 bytes")
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            signature, DOMAIN + _canonical(payload)
        )
    except InvalidSignature as exc:
        raise ValueError("receipt signature is invalid") from exc
    return json.loads(_canonical(payload))
