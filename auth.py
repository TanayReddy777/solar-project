"""HTTP Basic authentication and role authorization for protected CRUD APIs.

The application rejects Basic credentials on non-HTTPS requests. In production,
the ASGI server must trust the HTTPS reverse proxy's forwarded-proto header.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

PBKDF2_ITERATIONS = 600_000
ROLES = {"viewer", "operator", "admin"}
ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}
basic = HTTPBasic(auto_error=False, realm="Solar API")


def hash_password(password: str) -> str:
    """Create a salted PBKDF2-SHA256 password hash for API_USERS_JSON."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${encode(salt)}${encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds_text, salt_text, digest_text = encoded.split("$")
        rounds = int(rounds_text)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= rounds <= 2_000_000:
            return False
        decode = lambda value: base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), decode(salt_text), rounds
        )
        expected = decode(digest_text)
        return hmac.compare_digest(actual, expected)
    except (AttributeError, TypeError, ValueError):
        return False


def load_users() -> dict:
    raw = os.getenv("API_USERS_JSON")
    if not raw:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    try:
        users = json.loads(raw)
        if not isinstance(users, dict) or not users:
            raise ValueError("expected non-empty object")
        for username, user in users.items():
            if (
                not isinstance(username, str)
                or not username
                or not isinstance(user, dict)
                or not isinstance(user.get("password_hash"), str)
                or user.get("role") not in ROLES
            ):
                raise ValueError("invalid user record")
        return users
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(status_code=503, detail="Authentication user configuration is invalid")


def current_user(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials | None, Depends(basic)],
) -> dict:
    if request.url.scheme != "https":
        raise HTTPException(status_code=400, detail="HTTPS is required for Basic authentication")
    if credentials is None or len(credentials.password) > 1024:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": 'Basic realm="Solar API"'},
        )

    user = load_users().get(credentials.username)
    if user is None or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": 'Basic realm="Solar API"'},
        )
    return {"username": credentials.username, "role": user["role"]}


def require_role(minimum_role: str):
    if minimum_role not in ROLE_LEVEL:
        raise ValueError("Unknown role")

    def dependency(user: Annotated[dict, Depends(current_user)]) -> dict:
        if ROLE_LEVEL[user["role"]] < ROLE_LEVEL[minimum_role]:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return dependency
