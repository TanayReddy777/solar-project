# Solar power prediction API

Predicts plant AC power from weather and time, and flags unusual readings.
Model: SVR (scikit-learn 1.8.0). Hosted on Render.

Live URL: https://solar-api-876m.onrender.com
Interactive docs: https://solar-api-876m.onrender.com/docs

## MongoDB setup

The API stores forecast history and actual readings in MongoDB collections named
`forecasts` and `readings` inside the database selected by `MONGODB_DB` (defaults
to `solar_db`). Set `MONGODB_URI` in the API process environment. For MongoDB
Atlas, use the Python driver connection string and an Atlas database user. Never
put the URI in frontend code or commit it to Git.

PowerShell (local development):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MONGODB_URI = "mongodb+srv://<db_user>:<encoded_password>@<cluster-host>/?retryWrites=true&w=majority"
uvicorn app:app --reload
```

Set the same `MONGODB_URI` as an environment variable in the Render service before
using the hosted API. Restrict Atlas Network Access to the addresses required by
your development machine and hosting service. Do not use `0.0.0.0/0` for a
long-lived deployment.

## Endpoints
- GET /           health check
- POST /predict   predicts AC power and saves a forecast history record
- POST /deviation compares real and predicted power, flags unusual readings
- POST, GET, GET by id, PUT, DELETE `/readings`   CRUD for actual readings
- POST, GET, GET by id, PUT, DELETE `/forecasts`  CRUD for saved forecasts
- GET /docs       interactive API documentation

`POST /predict` stores input conditions, predicted power, and timestamps in
`forecasts`. `POST /readings` stores an actual reading together with predicted
power and deviation assessment. Forecasts created with `POST /forecasts` can be
used for what-if planning and are also stored in `forecasts`.

Run the database-independent API persistence checks with:

```powershell
python -m unittest discover -s tests -v
```

## Example request (/predict)
{
  "plant_id": 4136001,
  "date_time": "13-06-2020 13:30",
  "ambient_temperature": 31.9102,
  "module_temperature": 47.1552,
  "irradiation": 0.7514
}

Add "ac_power" to the same body for /deviation.
Valid plant_id values: 4135001, 4136001.

