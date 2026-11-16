import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo.errors import PyMongoError

import db

# ------------------------------------------------------------------ model + config
BASE = Path(__file__).parent
MODEL_DIR = BASE / "model" if (BASE / "model").exists() else BASE
model = joblib.load(MODEL_DIR / "solar_model.joblib")
config = json.loads((MODEL_DIR / "model_config.json").read_text())

FEATURES = config["features"]
THRESHOLDS = {str(k): float(v) for k, v in config["deviation_thresholds_kw"].items()}
CENTERS = {str(k): float(v) for k, v in config.get("residual_center_kw", {}).items()}
REL = config.get("relative_rule", {"ratio": 0.5, "min_predicted_kw": 2000})
PLANT_2_ID = int(config["plant_2_id"])        # plant_code = 1 for this plant, 0 for the other
DAY_IRR = float(config["daytime_irradiation"])

app = FastAPI(title="Solar power API", version="2.0")

# CORS: lets a web page on another address call this API with fetch()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your frontend URL later
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PyMongoError)
def database_error(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Database is not reachable. Check MONGODB_URI."})


# ------------------------------------------------------------------ request shapes
class Conditions(BaseModel):
    plant_id: int
    date_time: str
    ambient_temperature: float
    module_temperature: float
    irradiation: float


class Reading(Conditions):
    ac_power: float | None = None      # needed only for /deviation


class ReadingCreate(Conditions):
    ac_power: float
    note: str = ""


class ReadingUpdate(BaseModel):
    plant_id: int | None = None
    date_time: str | None = None
    ambient_temperature: float | None = None
    module_temperature: float | None = None
    irradiation: float | None = None
    ac_power: float | None = None
    note: str | None = None


class ForecastCreate(Conditions):
    note: str = ""


class ForecastUpdate(BaseModel):
    plant_id: int | None = None
    date_time: str | None = None
    ambient_temperature: float | None = None
    module_temperature: float | None = None
    irradiation: float | None = None
    note: str | None = None


# ------------------------------------------------------------------ model helpers
def parse_ts(value: str) -> pd.Timestamp:
    """Accept the dataset's formats (day-first and ISO)."""
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(value, dayfirst=True)
    except Exception:
        raise HTTPException(422, f"Unrecognised timestamp: {value}")


def check_plant(plant_id: int):
    if str(plant_id) not in THRESHOLDS:
        raise HTTPException(422, f"Unknown plant_id {plant_id}. Use one of {list(THRESHOLDS)}")


def build_row(r: Conditions) -> pd.DataFrame:
    check_plant(r.plant_id)
    ts = parse_ts(r.date_time)
    hour = ts.hour + ts.minute / 60
    row = {
        "plant_code": 1 if r.plant_id == PLANT_2_ID else 0,
        "AMBIENT_TEMPERATURE": r.ambient_temperature,
        "MODULE_TEMPERATURE": r.module_temperature,
        "IRRADIATION": r.irradiation,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
    }
    return pd.DataFrame([row])[FEATURES]


def predict_kw(r: Conditions) -> float:
    return max(float(model.predict(build_row(r))[0]), 0.0)


def assess(r: Conditions, ac_power: float) -> dict:
    """Compare an actual reading with the model's expectation."""
    pid = str(r.plant_id)
    pred = predict_kw(r)
    residual = ac_power - pred
    thr = THRESHOLDS[pid]
    daytime = r.irradiation > DAY_IRR

    # Rule 1: miss larger than the plant's threshold (after centring on its typical residual)
    size_rule = daytime and abs(residual - CENTERS.get(pid, 0.0)) > thr
    # Rule 2: output far below expected (catches shutdown-type drops at any size)
    relative_rule = (daytime
                     and pred > REL["min_predicted_kw"]
                     and ac_power < REL["ratio"] * pred)

    result = {
        "plant_id": r.plant_id,
        "actual_ac_power_kw": ac_power,
        "predicted_ac_power_kw": round(pred, 1),
        "residual_kw": round(residual, 1),
        "threshold_kw": round(thr, 1),
        "daytime": daytime,
        "size_rule": bool(size_rule),
        "relative_rule": bool(relative_rule),
        "deviation": bool(size_rule or relative_rule),
    }
    if not daytime:
        result["reason"] = "night-time reading, not checked"
    return result


# ------------------------------------------------------------------ database helpers
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_oid(id_text: str) -> ObjectId:
    if not ObjectId.is_valid(id_text):
        raise HTTPException(422, "Invalid id")
    return ObjectId(id_text)


def out(doc: dict) -> dict:
    """MongoDB's _id is an ObjectId; send it to the browser as a plain string called id."""
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return doc


def fetch_or_404(collection, id_text: str) -> dict:
    doc = collection.find_one({"_id": to_oid(id_text)})
    if doc is None:
        raise HTTPException(404, "Not found")
    return doc


def conditions_of(doc: dict) -> Conditions:
    return Conditions(**{k: doc[k] for k in Conditions.model_fields})


# ------------------------------------------------------------------ model endpoints
@app.get("/")
def health():
    return {"status": "ok", "endpoints": ["/predict", "/deviation", "/readings", "/forecasts", "/docs"]}


@app.post("/predict")
def predict(r: Reading):
    return {"plant_id": r.plant_id, "predicted_ac_power_kw": round(predict_kw(r), 1)}


@app.post("/deviation")
def deviation(r: Reading):
    if r.ac_power is None:
        raise HTTPException(422, "ac_power is required for /deviation")
    return assess(r, r.ac_power)


# ------------------------------------------------------------------ CRUD: readings (Plant Engineer)
@app.post("/readings", status_code=201)                      # CREATE
def create_reading(r: ReadingCreate):
    doc = r.model_dump()
    doc.update(assess(r, r.ac_power))
    doc["created_at"] = doc["updated_at"] = now()
    doc["_id"] = db.readings.insert_one(doc).inserted_id
    return out(doc)


@app.get("/readings")                                        # READ (list, with filters)
def list_readings(plant_id: int | None = None, deviation: bool | None = None,
                  limit: int = Query(50, ge=1, le=200), skip: int = Query(0, ge=0)):
    query = {}
    if plant_id is not None:
        query["plant_id"] = plant_id
    if deviation is not None:
        query["deviation"] = deviation
    cursor = db.readings.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return {"total": db.readings.count_documents(query), "items": [out(d) for d in cursor]}


@app.get("/readings/{reading_id}")                           # READ (one)
def get_reading(reading_id: str):
    return out(fetch_or_404(db.readings, reading_id))


@app.put("/readings/{reading_id}")                           # UPDATE
def update_reading(reading_id: str, changes: ReadingUpdate):
    doc = fetch_or_404(db.readings, reading_id)
    new = {**doc, **changes.model_dump(exclude_unset=True)}
    new.update(assess(conditions_of(new), new["ac_power"]))   # re-run the model on the edited values
    new["updated_at"] = now()
    new.pop("_id")
    db.readings.update_one({"_id": doc["_id"]}, {"$set": new})
    return out({**new, "_id": doc["_id"]})


@app.delete("/readings/{reading_id}")                        # DELETE
def delete_reading(reading_id: str):
    if db.readings.delete_one({"_id": to_oid(reading_id)}).deleted_count == 0:
        raise HTTPException(404, "Not found")
    return {"deleted": reading_id}


# ------------------------------------------------------------------ CRUD: forecasts (DISCOM)
@app.post("/forecasts", status_code=201)                     # CREATE
def create_forecast(f: ForecastCreate):
    doc = f.model_dump()
    doc["predicted_ac_power_kw"] = round(predict_kw(f), 1)
    doc["created_at"] = doc["updated_at"] = now()
    doc["_id"] = db.forecasts.insert_one(doc).inserted_id
    return out(doc)


@app.get("/forecasts")                                       # READ (list)
def list_forecasts(plant_id: int | None = None,
                   limit: int = Query(50, ge=1, le=200), skip: int = Query(0, ge=0)):
    query = {} if plant_id is None else {"plant_id": plant_id}
    cursor = db.forecasts.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return {"total": db.forecasts.count_documents(query), "items": [out(d) for d in cursor]}


@app.get("/forecasts/{forecast_id}")                         # READ (one)
def get_forecast(forecast_id: str):
    return out(fetch_or_404(db.forecasts, forecast_id))


@app.put("/forecasts/{forecast_id}")                         # UPDATE
def update_forecast(forecast_id: str, changes: ForecastUpdate):
    doc = fetch_or_404(db.forecasts, forecast_id)
    new = {**doc, **changes.model_dump(exclude_unset=True)}
    new["predicted_ac_power_kw"] = round(predict_kw(conditions_of(new)), 1)
    new["updated_at"] = now()
    new.pop("_id")
    db.forecasts.update_one({"_id": doc["_id"]}, {"$set": new})
    return out({**new, "_id": doc["_id"]})


@app.delete("/forecasts/{forecast_id}")                      # DELETE
def delete_forecast(forecast_id: str):
    if db.forecasts.delete_one({"_id": to_oid(forecast_id)}).deleted_count == 0:
        raise HTTPException(404, "Not found")
    return {"deleted": forecast_id}
