"""API-key authentication and tenant scope authorization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .control_db import ControlDB


DEFAULT_PBKDF2_ITERATIONS = 310_000


class AuthenticationError(Exception):
    """Raised when an API key is absent, invalid, or revoked."""


class AuthorizationError(Exception):
    """Raised when a key cannot act for a tenant or scope."""


class TokenIdempotencyConflict(Exception):
    """Raised when a Token issuance key is reused for another request."""


@dataclass(frozen=True)
class IssuedAPIKey:
    key_id: str
    tenant_id: str
    api_key: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class IssuedScopeToken:
    token_id: str
    tenant_id: str
    access_token: str
    permissions: frozenset[str]
    scope_names: frozenset[str]
    scope_prefixes: frozenset[str]
    label: str
    subject: str | None
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class AuthContext:
    key_id: str
    tenant_id: str
    scopes: frozenset[str]
    credential_type: str = "api_key"
    allowed_scope_names: frozenset[str] | None = None
    subject: str | None = None
    expires_at: float | None = None
    allowed_scope_prefixes: frozenset[str] | None = None

    def allows(self, scope: str) -> bool:
        return scope in self.scopes

    def allows_scope_name(self, scope_name: str) -> bool:
        if self.allowed_scope_names is None and self.allowed_scope_prefixes is None:
            return True
        return bool(
            scope_name in (self.allowed_scope_names or ())
            or any(
                scope_name.startswith(prefix)
                for prefix in (self.allowed_scope_prefixes or ())
            )
        )

    @property
    def credential_id(self) -> str:
        return self.key_id


def _normalize_scopes(scopes: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(str(scope).strip() for scope in scopes)
    if any(not scope for scope in normalized):
        raise ValueError("scopes must be non-empty strings")
    return normalized


_SCOPE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _normalize_scope_names(
    scope_names: Iterable[str], *, allow_empty: bool = False
) -> frozenset[str]:
    normalized = frozenset(str(scope_name).strip() for scope_name in scope_names)
    if (not normalized and not allow_empty) or any(
        not _SCOPE_NAME_RE.fullmatch(value) for value in normalized
    ):
        raise ValueError("scope_names must contain valid TMCRA scope names")
    return normalized


def _normalize_scope_prefixes(scope_prefixes: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(str(prefix).strip() for prefix in scope_prefixes)
    if any(not _SCOPE_NAME_RE.fullmatch(value) for value in normalized):
        raise ValueError("scope_prefixes must contain valid TMCRA scope prefixes")
    return normalized


def hash_api_key(api_key: str, *, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> str:
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("api_key must be a non-empty string")
    if iterations < 100_000:
        raise ValueError("iterations is too low")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", api_key.encode("utf-8"), salt, iterations)
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${iterations}${encode(salt)}${encode(digest)}"


def verify_api_key(api_key: str, encoded_hash: str) -> bool:
    try:
        algorithm, iteration_text, salt_text, digest_text = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iteration_text)
        decode = lambda value: base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        salt = decode(salt_text)
        expected = decode(digest_text)
        actual = hashlib.pbkdf2_hmac("sha256", api_key.encode("utf-8"), salt, iterations)
    except (AttributeError, TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, expected)


def _decode_derivation_key(value: str) -> bytes:
    try:
        decoded = base64.urlsafe_b64decode(value.strip() + "=" * (-len(value.strip()) % 4))
    except (ValueError, TypeError) as exc:
        raise ValueError("TMCRA_SCOPE_TOKEN_DERIVATION_KEY is not valid base64url") from exc
    if len(decoded) != 32:
        raise ValueError("TMCRA_SCOPE_TOKEN_DERIVATION_KEY must decode to 32 bytes")
    return decoded


def _load_or_create_token_derivation_key(db: ControlDB) -> bytes:
    configured = os.getenv("TMCRA_SCOPE_TOKEN_DERIVATION_KEY", "").strip()
    if configured:
        return _decode_derivation_key(configured)
    if db.path == ":memory:":
        return secrets.token_bytes(32)

    key_path = Path(f"{db.path}.scope-token-key")
    try:
        return _decode_derivation_key(key_path.read_text(encoding="ascii"))
    except FileNotFoundError:
        pass

    encoded = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _decode_derivation_key(key_path.read_text(encoding="ascii"))
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(f"{encoded}\n")
    except BaseException:
        key_path.unlink(missing_ok=True)
        raise
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return _decode_derivation_key(encoded)


class APIKeyAuth:
    """Issue and validate keys without persisting raw key material."""

    def __init__(
        self,
        db: ControlDB,
        *,
        iterations: int = DEFAULT_PBKDF2_ITERATIONS,
        token_derivation_key: bytes | None = None,
    ) -> None:
        self.db = db
        self.iterations = iterations
        self._token_derivation_key = (
            bytes(token_derivation_key)
            if token_derivation_key is not None
            else _load_or_create_token_derivation_key(db)
        )
        if len(self._token_derivation_key) != 32:
            raise ValueError("token_derivation_key must contain exactly 32 bytes")
        self._backfill_scope_token_replay_hashes()

    def _backfill_scope_token_replay_hashes(self) -> None:
        with self.db.transaction() as connection:
            rows = connection.execute(
                """
                SELECT i.token_id,t.secret_hash
                FROM scope_token_issuances AS i
                JOIN scope_tokens AS t ON t.token_id=i.token_id
                WHERE i.token_replay_hash IS NULL
                """
            ).fetchall()
            for row in rows:
                access_token = self._derived_scope_token(str(row["token_id"]))
                if not verify_api_key(access_token, str(row["secret_hash"])):
                    raise RuntimeError(
                        "scope Token derivation key no longer matches persisted issuances"
                    )
                connection.execute(
                    "UPDATE scope_token_issuances SET token_replay_hash=? WHERE token_id=?",
                    (
                        hashlib.sha256(access_token.encode("utf-8")).hexdigest(),
                        str(row["token_id"]),
                    ),
                )

    def set_tenant_scopes(self, tenant_id: str, scopes: Iterable[str]) -> None:
        self.db.set_tenant_scopes(tenant_id, _normalize_scopes(scopes))

    def create_key(self, tenant_id: str, scopes: Iterable[str] | None = None) -> IssuedAPIKey:
        allowed = self.db.get_tenant_scopes(tenant_id)
        requested = allowed if scopes is None else _normalize_scopes(scopes)
        if not requested <= allowed:
            raise AuthorizationError("key scopes exceed the tenant scope mapping")
        key_id = secrets.token_hex(12)
        raw_key = f"tmcra_{key_id}.{secrets.token_urlsafe(32)}"
        secret_hash = hash_api_key(raw_key, iterations=self.iterations)
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO api_keys
                    (key_id, tenant_id, secret_hash, scopes_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key_id, tenant_id, secret_hash, self.db.encode_json(sorted(requested)), time.time()),
            )
        return IssuedAPIKey(key_id, tenant_id, raw_key, requested)

    issue_key = create_key

    def create_scope_token(
        self,
        parent: AuthContext,
        *,
        permissions: Iterable[str],
        scope_names: Iterable[str] = (),
        label: str,
        subject: str | None,
        expires_at: float,
        scope_prefixes: Iterable[str] = (),
        idempotency_key: str | None = None,
        expires_in_seconds: int | None = None,
        provisional_delivery_seconds: int | None = None,
    ) -> IssuedScopeToken:
        if parent.credential_type != "api_key":
            raise AuthorizationError("only an API key may issue scoped access tokens")
        requested_permissions = _normalize_scopes(permissions)
        requested_scope_names = _normalize_scope_names(scope_names, allow_empty=True)
        requested_scope_prefixes = _normalize_scope_prefixes(scope_prefixes)
        if not requested_scope_names and not requested_scope_prefixes:
            raise ValueError("at least one scope name or scope prefix is required")
        tenant_permissions = self.db.get_tenant_scopes(parent.tenant_id)
        if not requested_permissions <= parent.scopes or not requested_permissions <= tenant_permissions:
            raise AuthorizationError("token permissions exceed the issuing key or tenant policy")
        forbidden = {
            "tokens:manage",
            "webhooks:manage",
            "retention:manage",
            "memory:delete",
            "memory:export",
        }
        if requested_permissions & forbidden:
            raise AuthorizationError("terminal access tokens cannot receive administrative permissions")
        clean_label = str(label).strip()
        clean_subject = None if subject is None else str(subject).strip()
        if not clean_label or len(clean_label) > 120:
            raise ValueError("label must be 1-120 characters")
        if clean_subject is not None and (not clean_subject or len(clean_subject) > 200):
            raise ValueError("subject must be 1-200 characters when provided")
        now = time.time()
        if expires_at <= now or expires_at > now + 366 * 86_400:
            raise ValueError("expires_at must be within the next 366 days")
        if idempotency_key is not None:
            clean_idempotency_key = str(idempotency_key).strip()
            if not 8 <= len(clean_idempotency_key) <= 200:
                raise ValueError("idempotency_key must be 8-200 characters")
            if expires_in_seconds is None or not 60 <= expires_in_seconds <= 366 * 86_400:
                raise ValueError("expires_in_seconds must be 60 seconds to 366 days")
            payload = {
                "permissions": sorted(requested_permissions),
                "scope_names": sorted(requested_scope_names),
                "scope_prefixes": sorted(requested_scope_prefixes),
                "label": clean_label,
                "subject": clean_subject,
                "expires_in_seconds": int(expires_in_seconds),
                "provisional_delivery_seconds": provisional_delivery_seconds,
            }
            payload_hash = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            with self.db.transaction() as connection:
                existing = connection.execute(
                    """
                    SELECT i.payload_hash,i.token_replay_hash,t.token_id,t.tenant_id,
                           t.permissions_json,t.revoked_at,
                           t.scope_names_json,t.scope_prefixes_json,t.label,t.subject,
                           t.created_at,t.expires_at
                    FROM scope_token_issuances AS i
                    JOIN scope_tokens AS t ON t.token_id=i.token_id
                    WHERE i.tenant_id=? AND i.created_by_key_id=?
                      AND i.idempotency_key=?
                    """,
                    (parent.tenant_id, parent.key_id, clean_idempotency_key),
                ).fetchone()
                if existing is not None:
                    if not hmac.compare_digest(str(existing["payload_hash"]), payload_hash):
                        raise TokenIdempotencyConflict(
                            "Idempotency-Key was already used with a different Token request"
                        )
                    if existing["revoked_at"] is not None:
                        raise TokenIdempotencyConflict(
                            "Idempotency-Key refers to a revoked Token; use a new key"
                        )
                    if float(existing["expires_at"]) <= time.time():
                        raise TokenIdempotencyConflict(
                            "Idempotency-Key refers to an expired Token; use a new key"
                        )
                    return self._issued_scope_token_from_row(existing)

                token_id = secrets.token_hex(12)
                raw_token = self._derived_scope_token(token_id)
                created_at = time.time()
                final_expires_at = created_at + int(expires_in_seconds)
                stable_expires_at = (
                    min(final_expires_at, created_at + int(provisional_delivery_seconds))
                    if provisional_delivery_seconds is not None
                    else final_expires_at
                )
                secret_hash = hash_api_key(raw_token, iterations=self.iterations)
                connection.execute(
                    """
                    INSERT INTO scope_tokens(
                        token_id,tenant_id,secret_hash,permissions_json,scope_names_json,
                        scope_prefixes_json,label,subject,created_by_key_id,created_at,expires_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        token_id,
                        parent.tenant_id,
                        secret_hash,
                        self.db.encode_json(sorted(requested_permissions)),
                        self.db.encode_json(sorted(requested_scope_names)),
                        self.db.encode_json(sorted(requested_scope_prefixes)),
                        clean_label,
                        clean_subject,
                        parent.key_id,
                        created_at,
                        stable_expires_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO scope_token_issuances(
                        tenant_id,created_by_key_id,idempotency_key,payload_hash,
                        token_id,token_replay_hash,final_expires_at,confirmed_at,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        parent.tenant_id,
                        parent.key_id,
                        clean_idempotency_key,
                        payload_hash,
                        token_id,
                        hashlib.sha256(raw_token.encode("utf-8")).hexdigest(),
                        final_expires_at,
                        None if provisional_delivery_seconds is not None else created_at,
                        created_at,
                    ),
                )
                return IssuedScopeToken(
                    token_id=token_id,
                    tenant_id=parent.tenant_id,
                    access_token=raw_token,
                    permissions=requested_permissions,
                    scope_names=requested_scope_names,
                    scope_prefixes=requested_scope_prefixes,
                    label=clean_label,
                    subject=clean_subject,
                    created_at=created_at,
                    expires_at=stable_expires_at,
                )

        token_id = secrets.token_hex(12)
        raw_token = f"tmcra_st_{token_id}.{secrets.token_urlsafe(32)}"
        secret_hash = hash_api_key(raw_token, iterations=self.iterations)
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scope_tokens(
                    token_id,tenant_id,secret_hash,permissions_json,scope_names_json,
                    scope_prefixes_json,label,subject,created_by_key_id,created_at,expires_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    token_id,
                    parent.tenant_id,
                    secret_hash,
                    self.db.encode_json(sorted(requested_permissions)),
                    self.db.encode_json(sorted(requested_scope_names)),
                    self.db.encode_json(sorted(requested_scope_prefixes)),
                    clean_label,
                    clean_subject,
                    parent.key_id,
                    now,
                    float(expires_at),
                ),
            )
        return IssuedScopeToken(
            token_id=token_id,
            tenant_id=parent.tenant_id,
            access_token=raw_token,
            permissions=requested_permissions,
            scope_names=requested_scope_names,
            scope_prefixes=requested_scope_prefixes,
            label=clean_label,
            subject=clean_subject,
            created_at=now,
            expires_at=float(expires_at),
        )

    def confirm_scope_token(
        self,
        parent: AuthContext,
        token_id: str,
    ) -> dict[str, object] | None:
        if parent.credential_type != "api_key":
            raise AuthorizationError("only an API key may confirm scoped access tokens")
        now = time.time()
        with self.db.transaction() as connection:
            row = connection.execute(
                """
                SELECT i.final_expires_at,i.confirmed_at,t.token_id,t.tenant_id,
                       t.permissions_json,t.scope_names_json,t.scope_prefixes_json,
                       t.label,t.subject,t.created_by_key_id,t.created_at,t.expires_at,
                       t.revoked_at,t.last_used_at
                FROM scope_token_issuances AS i
                JOIN scope_tokens AS t ON t.token_id=i.token_id
                WHERE i.tenant_id=? AND i.created_by_key_id=? AND i.token_id=?
                """,
                (parent.tenant_id, parent.key_id, token_id),
            ).fetchone()
            if row is None:
                return None
            if row["revoked_at"] is not None:
                raise AuthorizationError("revoked access tokens cannot be confirmed")
            if row["confirmed_at"] is None and float(row["expires_at"]) <= now:
                raise AuthorizationError("provisional access token expired before confirmation")
            final_expires_at = float(row["final_expires_at"])
            if row["confirmed_at"] is None:
                connection.execute(
                    "UPDATE scope_tokens SET expires_at=? WHERE token_id=?",
                    (final_expires_at, token_id),
                )
                connection.execute(
                    "UPDATE scope_token_issuances SET confirmed_at=? WHERE token_id=?",
                    (now, token_id),
                )
            return {
                "token_id": str(row["token_id"]),
                "tenant_id": str(row["tenant_id"]),
                "permissions": json.loads(str(row["permissions_json"])),
                "scope_names": json.loads(str(row["scope_names_json"])),
                "scope_prefixes": json.loads(str(row["scope_prefixes_json"])),
                "label": str(row["label"]),
                "subject": row["subject"],
                "created_by_key_id": str(row["created_by_key_id"]),
                "created_at": float(row["created_at"]),
                "expires_at": final_expires_at,
                "revoked_at": None,
                "last_used_at": (
                    None if row["last_used_at"] is None else float(row["last_used_at"])
                ),
            }

    def _derived_scope_token(self, token_id: str) -> str:
        secret = hmac.new(
            self._token_derivation_key,
            f"tmcra-scope-token-v1:{token_id}".encode("ascii"),
            hashlib.sha256,
        ).digest()
        encoded = base64.urlsafe_b64encode(secret).decode("ascii").rstrip("=")
        return f"tmcra_st_{token_id}.{encoded}"

    def _issued_scope_token_from_row(self, row: Any) -> IssuedScopeToken:
        token_id = str(row["token_id"])
        access_token = self._derived_scope_token(token_id)
        if "token_replay_hash" in row.keys() and not hmac.compare_digest(
            hashlib.sha256(access_token.encode("utf-8")).hexdigest(),
            str(row["token_replay_hash"]),
        ):
            raise RuntimeError(
                "scope Token derivation key no longer matches persisted issuances"
            )
        return IssuedScopeToken(
            token_id=token_id,
            tenant_id=str(row["tenant_id"]),
            access_token=access_token,
            permissions=frozenset(json.loads(str(row["permissions_json"]))),
            scope_names=frozenset(json.loads(str(row["scope_names_json"]))),
            scope_prefixes=frozenset(json.loads(str(row["scope_prefixes_json"]))),
            label=str(row["label"]),
            subject=None if row["subject"] is None else str(row["subject"]),
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )

    def list_scope_tokens(self, tenant_id: str) -> list[dict[str, object]]:
        with self.db.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT token_id,tenant_id,permissions_json,scope_names_json,
                       scope_prefixes_json,label,subject,
                       created_by_key_id,created_at,expires_at,revoked_at,last_used_at
                FROM scope_tokens WHERE tenant_id=? ORDER BY created_at,token_id
                """,
                (tenant_id,),
            ).fetchall()
        return [
            {
                "token_id": str(row["token_id"]),
                "tenant_id": str(row["tenant_id"]),
                "permissions": json.loads(str(row["permissions_json"])),
                "scope_names": json.loads(str(row["scope_names_json"])),
                "scope_prefixes": json.loads(str(row["scope_prefixes_json"])),
                "label": str(row["label"]),
                "subject": row["subject"],
                "created_by_key_id": str(row["created_by_key_id"]),
                "created_at": float(row["created_at"]),
                "expires_at": float(row["expires_at"]),
                "revoked_at": None if row["revoked_at"] is None else float(row["revoked_at"]),
                "last_used_at": None if row["last_used_at"] is None else float(row["last_used_at"]),
            }
            for row in rows
        ]

    def revoke_scope_token(self, tenant_id: str, token_id: str) -> bool:
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scope_tokens SET revoked_at=?
                WHERE token_id=? AND tenant_id=? AND revoked_at IS NULL
                """,
                (time.time(), token_id, tenant_id),
            )
        return cursor.rowcount == 1

    def revoke_key(self, key_id: str) -> bool:
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (time.time(), key_id),
            )
        return cursor.rowcount == 1

    def authenticate(self, api_key: str) -> AuthContext:
        if not isinstance(api_key, str) or not api_key:
            raise AuthenticationError("invalid API key")
        if api_key.startswith("tmcra_st_"):
            return self._authenticate_scope_token(api_key)
        key_id = api_key.split(".", 1)[0].removeprefix("tmcra_")
        with self.db.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT key_id, tenant_id, secret_hash, scopes_json
                FROM api_keys
                WHERE key_id = ? AND revoked_at IS NULL
                """,
                (key_id,),
            ).fetchone()
            if row is None:
                raise AuthenticationError("invalid API key")
        if not verify_api_key(api_key, row["secret_hash"]):
            raise AuthenticationError("invalid API key")
        scopes = frozenset(json.loads(row["scopes_json"]))
        return AuthContext(row["key_id"], row["tenant_id"], scopes)

    def _authenticate_scope_token(self, access_token: str) -> AuthContext:
        token_id = access_token.split(".", 1)[0].removeprefix("tmcra_st_")
        now = time.time()
        with self.db.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT token_id,tenant_id,secret_hash,permissions_json,scope_names_json,
                       scope_prefixes_json,subject,expires_at,last_used_at
                FROM scope_tokens
                WHERE token_id=? AND revoked_at IS NULL AND expires_at>?
                """,
                (token_id, now),
            ).fetchone()
        if row is None or not verify_api_key(access_token, str(row["secret_hash"])):
            raise AuthenticationError("invalid access token")
        last_used_at = row["last_used_at"]
        if last_used_at is None or float(last_used_at) < now - 900:
            with self.db.transaction() as connection:
                connection.execute(
                    """
                    UPDATE scope_tokens SET last_used_at=?
                    WHERE token_id=? AND revoked_at IS NULL
                    """,
                    (now, token_id),
                )
        return AuthContext(
            key_id=str(row["token_id"]),
            tenant_id=str(row["tenant_id"]),
            scopes=frozenset(json.loads(str(row["permissions_json"]))),
            credential_type="scope_token",
            allowed_scope_names=frozenset(json.loads(str(row["scope_names_json"]))),
            allowed_scope_prefixes=frozenset(
                json.loads(str(row["scope_prefixes_json"]))
            ),
            subject=row["subject"],
            expires_at=float(row["expires_at"]),
        )

    def authorize(
        self,
        api_key: str,
        tenant_id: str,
        required_scopes: Iterable[str] = (),
    ) -> AuthContext:
        context = self.authenticate(api_key)
        required = _normalize_scopes(required_scopes)
        if context.tenant_id != tenant_id:
            raise AuthorizationError("API key is not valid for this tenant")
        tenant_scopes = self.db.get_tenant_scopes(tenant_id)
        if not required <= context.scopes or not required <= tenant_scopes:
            raise AuthorizationError("scope is not granted by both key and tenant policy")
        return context
