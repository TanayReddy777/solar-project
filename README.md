# Solar power prediction API

Predicts plant AC power from weather and time, and flags unusual readings.
Model: SVR (scikit-learn 1.8.0). Hosted on Render.

Live URL: https://solar-api-876m.onrender.com
Interactive docs: https://solar-api-876m.onrender.com/docs

## Endpoints
- GET / health check
- POST /predict predicted AC power (kW), public for frontend compatibility
- POST /deviation compares real and predicted power
- POST /auth/token issues a 30-minute bearer token
- Readings and forecasts CRUD routes require bearer authentication

## Authentication configuration (Render)

Set these variables in Render's Environment settings. No secrets or user credentials have defaults.

- `JWT_SECRET_KEY`: a random secret of at least 32 UTF-8 bytes. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- `API_USERS_JSON`: a JSON object keyed by username; each value has a PBKDF2 password hash and role (`viewer`, `operator`, or `admin`). Example shape: `{"solar-viewer":{"password_hash":"<generated hash>","role":"viewer"}}`.

Generate a password hash without placing a plaintext password in command history:
`python -c "from getpass import getpass; from auth import hash_password; print(hash_password(getpass()))"`

Viewers can read. Operators can read and create. Admins can read, create, update, and delete. Tokens use HS256, issuer and audience claims, and a 30-minute expiry. User existence and role are checked on every request, so removing an account or changing its role invalidates existing tokens. Do not commit credentials or signing keys.

## Example prediction request
{
  "plant_id": 4136001,
  "date_time": "13-06-2020 13:30",
  "ambient_temperature": 31.9102,
  "module_temperature": 47.1552,
  "irradiation": 0.7514
}

Add `ac_power` to the same body for `/deviation`. Valid plant IDs: 4135001 and 4136001.

## Verification
Install development packages with `pip install -r requirements-dev.txt`, then run `python -m pytest -v` and `python -m compileall .`.
