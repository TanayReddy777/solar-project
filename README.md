# Solar power prediction API

Predicts plant AC power from weather and time, and flags unusual readings.
Model: SVR (scikit-learn 1.8.0). Hosted on Render.

Live URL: https://solar-api-876m.onrender.com
Interactive docs: https://solar-api-876m.onrender.com/docs

## Endpoints
- GET / health check
- POST /predict predicted AC power (kW), public for frontend compatibility
- POST /deviation compares real and predicted power
- Readings and forecasts CRUD routes require HTTP Basic authentication over HTTPS

## Authentication configuration (Render)

Set `API_USERS_JSON` in Render's Environment settings. It is a JSON object keyed by username; each value contains a salted PBKDF2-SHA256 password hash and one role (`viewer`, `operator`, or `admin`). Example shape:
`{"solar-viewer":{"password_hash":"<generated hash>","role":"viewer"}}`

Generate a password hash without placing plaintext in shell history:
`python -c "from getpass import getpass; from auth import hash_password; print(hash_password(getpass()))"`

Viewers can read. Operators can read and create. Admins can read, create, update, and delete. Passwords are never stored in the repository; configure only password hashes in the environment.

HTTP Basic credentials are accepted only for HTTPS requests. Render terminates TLS and forwards the original protocol to Uvicorn. Configure the Render start command to trust the Render proxy header, for example:
`uvicorn app:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips=*`
Use this setting only for a service that is reachable through Render's proxy. Verify the deployed service presents an HTTPS URL; do not send Basic credentials over plain HTTP. This change does not modify or trigger a deployment.

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
