"""Environment-configured JWT authentication for protected CRUD APIs."""
import base64, hashlib, hmac, json, os, secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated
import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

JWT_ISSUER = "solar-project-api"
JWT_AUDIENCE = "solar-project-clients"
TOKEN_TTL_SECONDS = 1800
PBKDF2_ITERATIONS = 600_000
ROLES = {"viewer", "operator", "admin"}
ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}
router = APIRouter(tags=["authentication"])
bearer = HTTPBearer(auto_error=False)

class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)

def hash_password(password: str) -> str:
    """Create a PBKDF2-SHA256 password hash for API_USERS_JSON."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    enc = lambda value: base64.urlsafe_b64encode(value).decode().rstrip("=")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${enc(salt)}${enc(digest)}"

def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_text, digest_text = encoded.split("$")
        rounds = int(rounds)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= rounds <= 2_000_000:
            return False
        dec = lambda value: base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        expected = dec(digest_text)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), dec(salt_text), rounds)
        return hmac.compare_digest(actual, expected)
    except (AttributeError, TypeError, ValueError):
        return False

def load_users() -> dict:
    raw = os.getenv("API_USERS_JSON")
    if not raw:
        raise HTTPException(503, "Authentication is not configured")
    try:
        users = json.loads(raw)
        if not isinstance(users, dict) or not users:
            raise ValueError("expected non-empty object")
        for name, user in users.items():
            if (not isinstance(name, str) or not name or not isinstance(user, dict)
                or not isinstance(user.get("password_hash"), str)
                or user.get("role") not in ROLES):
                raise ValueError("invalid user record")
        return users
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(503, "Authentication user configuration is invalid")

def signing_key() -> str:
    key = os.getenv("JWT_SECRET_KEY", "")
    if len(key.encode()) < 32:
        raise HTTPException(503, "JWT signing key is missing or too short")
    return key

@router.post("/auth/token")
def issue_token(request: TokenRequest):
    user = load_users().get(request.username)
    if user is None or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(401, "Invalid username or password")
    now = datetime.now(timezone.utc)
    encoded = jwt.encode(
        {"sub": request.username, "role": user["role"], "iat": now,
         "exp": now + timedelta(seconds=TOKEN_TTL_SECONDS),
         "iss": JWT_ISSUER, "aud": JWT_AUDIENCE},
        signing_key(), algorithm="HS256",
    )
    return {"access_token": encoded, "token_type": "bearer", "expires_in": TOKEN_TTL_SECONDS}

def current_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = jwt.decode(credentials.credentials, signing_key(), algorithms=["HS256"],
            issuer=JWT_ISSUER, audience=JWT_AUDIENCE,
            options={"require": ["exp", "iat", "sub", "role", "iss", "aud"]})
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    user = load_users().get(claims.get("sub"))
    if user is None or claims.get("role") != user["role"]:
        raise HTTPException(401, "Token user is no longer authorized")
    return {"username": claims["sub"], "role": user["role"]}

def require_role(minimum_role: str):
    if minimum_role not in ROLE_LEVEL:
        raise ValueError("Unknown role")
    def dependency(user: Annotated[dict, Depends(current_user)]) -> dict:
        if ROLE_LEVEL[user["role"]] < ROLE_LEVEL[minimum_role]:
            raise HTTPException(403, "Insufficient role")
        return user
    return dependency
