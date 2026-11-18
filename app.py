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
import logging


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


class Evidence(BaseModel):
    inverters_reporting: int | None = None
    inverters_zero_ac: int | None = None
    previous_irradiation: float | None = None

class Reading(Conditions, Evidence):
    ac_power: float | None = None

class ReadingCreate(Conditions, Evidence):
    ac_power: float
    note: str = ""


class Evidence(BaseModel):
    inverters_reporting: int | None = None
    inverters_zero_ac: int | None = None
    previous_irradiation: float | None = None

class Reading(Conditions, Evidence):
    ac_power: float | None = None        # needed only for /deviation

class ReadingCreate(Conditions, Evidence):
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
    inverters_reporting: int | None = None     # new
    inverters_zero_ac: int | None = None       # new
    previous_irradiation: float | None = None  # new


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


FLEET = int(config.get("inverters_per_plant", 22))
CLOUD_JUMP = float(config.get("cloud_irradiation_jump", 0.15))


def evidence_of(obj) -> dict:
    """Read the optional evidence from a pydantic model or from a stored document."""
    get = obj.get if isinstance(obj, dict) else (lambda k: getattr(obj, k, None))
    return {k: get(k) for k in Evidence.model_fields}


def diagnose(r: Conditions, ac_power: float, pred: float, residual: float,
             center: float, thr: float, daytime: bool, deviation: bool,
             ev: dict | None = None) -> dict:
    ev = ev or {}
    n_rep, n_zero, prev_irr = ev.get("inverters_reporting"), ev.get("inverters_zero_ac"), ev.get("previous_irradiation")
    has_inverter_data = n_rep is not None or n_zero is not None

    def out(reason, causes, confidence, severity, lost=None, confirm=None):
        return {"reason": reason, "likely_causes": causes, "confidence": confidence,
                "severity": severity, "estimated_inverters_lost": lost,
                "to_confirm": confirm, "evidence_used": has_inverter_data or prev_irr is not None}

    if not daytime:
        return out("Night-time reading, not checked", [], "n/a", "none")
    if not deviation:
        return out("Normal: actual output is within the expected range", [], "n/a", "none")

    # how bad is it? (distance outside the allowed band, in units of the threshold)
    excess = abs(residual - center) / thr if thr else 0
    ratio = ac_power / pred if pred > 0 else None
    severity = "severe" if (excess > 2 or (ratio is not None and ratio < 0.5)) else "moderate"
    # shortfall expressed as "equivalent inverters" (fleet output is spread over FLEET inverters)
    lost = round((pred - ac_power) / pred * FLEET) if (residual < 0 and pred > 0) else None
    lost = min(lost, FLEET) if lost is not None else None
    cloud = prev_irr is not None and abs(r.irradiation - prev_irr) > CLOUD_JUMP

    # ---------- A) inverter-level evidence supplied -> diagnosis ----------
    if has_inverter_data:
        if n_rep is not None and n_rep < FLEET and not n_zero:
            return out(f"Only {n_rep} of {FLEET} inverters reported data, so the total is understated. "
                       "This is a data-logging gap, not necessarily a real loss of generation.",
                       ["Missing inverter data"], "high", severity, lost,
                       "Check the data logger / communication for the missing inverters")
        if n_zero is not None and n_zero > 0:
            total = n_rep if n_rep is not None else FLEET
            if n_zero >= total:
                return out("All inverters report 0 kW while sunlight is available: full plant outage "
                           "(grid trip / shutdown) or a data-logger fault.",
                           ["Full plant outage", "Data-logger fault"], "high", severity, lost,
                           "Check grid-trip and SCADA logs for this timestamp")
            shortfall = pred - ac_power
            explained = (n_zero / FLEET * pred) / shortfall if shortfall > 0 else None
            note = ""
            if explained is not None and explained < 0.7:
                note = (f" They explain only about {explained:.0%} of the shortfall, so the remaining "
                        "inverters are also under-producing.")
            return out(f"{n_zero} of {total} inverters are at 0 kW: partial inverter outage.{note}",
                       ["Partial inverter outage"] + (["Other inverters under-producing"] if note else []),
                       "high", severity, lost, "Inspect the inverters that show 0 kW")
        if residual < 0:
            causes = ["Plant-wide derating or curtailment", "Soiling / shading", "Irradiation sensor over-reading"]
            if cloud:
                causes.insert(0, "Sudden cloud cover")
            return out("No inverter is at 0 kW but output is low across the plant: "
                       "plant-wide derating, soiling, shading or a sensor mismatch.",
                       causes, "medium", severity, lost,
                       "Compare each inverter's output with its neighbours and check the weather sensor")

    # ---------- B) single reading only -> inference ----------
    if residual > 0:
        causes = ["Cloud-edge / irradiation sensor timing", "Irradiation sensor under-reading"]
        return out("Output is above what the weather inputs predict: usually a cloud-edge effect or "
                   "irradiation sensor lag.", causes, "low", severity, None,
                   "Check the previous and next readings")
    if ac_power == 0:
        return out("Output is exactly 0 kW although irradiation shows sunlight: most likely a full plant "
                   "outage or a data-logger fault.",
                   ["Full plant outage", "Data-logger fault"], "medium", severity, lost,
                   "Send inverters_zero_ac / inverters_reporting to confirm")
    causes = ["Partial inverter outage", "Plant-wide derating",
              "Sudden cloud cover (irradiation changed sharply)" if cloud else "Cloud cover",
              "Missing inverter data"]
    if cloud:
        causes.insert(0, causes.pop(2))      # most likely cause first
    approx = f" The shortfall equals roughly {lost} of {FLEET} inverters." if lost else ""
    return out(f"Output is {abs(residual):.0f} kW below expectation.{approx} "
               "One reading cannot separate the causes listed.",
               causes, "low", severity, lost,
               "Send inverters_zero_ac / inverters_reporting / previous_irradiation to confirm")


def assess(r: Conditions, ac_power: float, evidence: dict | None = None) -> dict:
    """Compare an actual reading with the model's expectation and explain the gap."""
    pid = str(r.plant_id)
    pred = predict_kw(r)
    residual = ac_power - pred
    thr = THRESHOLDS[pid]
    center = CENTERS.get(pid, 0.0)
    daytime = r.irradiation > DAY_IRR

    size_rule = daytime and abs(residual - center) > thr
    relative_rule = (daytime and pred > REL["min_predicted_kw"]
                     and ac_power < REL["ratio"] * pred)
    deviation = bool(size_rule or relative_rule)

    result = {
        "plant_id": r.plant_id,
        "actual_ac_power_kw": ac_power,
        "predicted_ac_power_kw": round(pred, 1),
        "residual_kw": round(residual, 1),
        "residual_centre_kw": round(center, 1),
        "threshold_kw": round(thr, 1),
        "daytime": daytime,
        "size_rule": bool(size_rule),
        "relative_rule": bool(relative_rule),
        "deviation": deviation,
        "calculation": (f"residual = actual - predicted = {ac_power:.1f} - {pred:.1f} = {residual:.1f} kW; "
                        f"allowed band = {center:.1f} +/- {thr:.1f} kW"),
    }
    result.update(diagnose(r, ac_power, pred, residual, center, thr, daytime, deviation, evidence))
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
@app.get("/health/db")
def health_db():
    db.client.admin.command("ping")
    return {"database": "connected"}



@app.post("/predict")
def predict(r: Reading):
    return {"plant_id": r.plant_id, "predicted_ac_power_kw": round(predict_kw(r), 1)}


@app.post("/deviation")
def deviation(r: Reading):
    if r.ac_power is None:
        raise HTTPException(422, "ac_power is required for /deviation")
        return assess(r, r.ac_power, evidence_of(r))


# ------------------------------------------------------------------ CRUD: readings (Plant Engineer)
@app.post("/readings", status_code=201)                      # CREATE
def create_reading(r: ReadingCreate):
    doc = r.model_dump()
    doc.update(assess(r, r.ac_power, evidence_of(r)))
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
   new.update(assess(conditions_of(new), new["ac_power"], evidence_of(new)))   # re-run the model on the edited values  # re-run the model on the edited values
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

logger = logging.getLogger("uvicorn.error")

@app.exception_handler(PyMongoError)
def database_error(request, exc):
    logger.error("MongoDB error: %r", exc)
    return JSONResponse(status_code=503, content={"detail": "Database is not reachable. Check MONGODB_URI."})
