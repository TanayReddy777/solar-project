import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from bson import ObjectId
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo.errors import PyMongoError

import db
from auth import require_role


# ------------------------------------------------------------------ model + config


BASE = Path(__file__).parent
MODEL_DIR = BASE / "model" if (BASE / "model").exists() else BASE
model = joblib.load(MODEL_DIR / "solar_model.joblib")
config = json.loads((MODEL_DIR / "model_config.json").read_text())

FEATURES = config["features"]
THRESHOLDS = {str(k): float(v) for k, v in config["deviation_thresholds_kw"].items()}
CENTERS = {str(k): float(v) for k, v in config.get("residual_center_kw", {}).items()}
REL = config.get("relative_rule")
PLANT_2_ID = int(config["plant_2_id"])        # plant_code = 1 for this plant, 0 for the other
DAY_IRR = float(config["daytime_irradiation"])
FLEET = int(config.get("inverters_per_plant", 22))
CLOUD_JUMP = float(config.get("cloud_irradiation_jump", 0.15))

app = FastAPI(title="Solar power API", version="2.1")

# CORS: lets a web page on another address call this API with fetch()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your frontend URL later
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger("uvicorn.error")


@app.exception_handler(PyMongoError)
def database_error(request, exc):
    logger.error("MongoDB error: %r", exc)
    return JSONResponse(status_code=503, content={"detail": "Database is not reachable. Check MONGODB_URI."})


# ------------------------------------------------------------------ request shapes
class Conditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plant_id: int
    date_time: str
    ambient_temperature: float
    module_temperature: float
    irradiation: float

    @field_validator("ambient_temperature", "module_temperature", "irradiation")
    @classmethod
    def validate_finite_weather(cls, value):
        if not np.isfinite(value):
            raise ValueError("Weather inputs must be finite numbers")
        return value

    @field_validator("irradiation")
    @classmethod
    def validate_nonnegative_irradiation(cls, value):
        if value < 0:
            raise ValueError("irradiation must be non-negative")
        return value


class Evidence(BaseModel):
    inverters_reporting: int | None = Field(default=None, ge=0, le=FLEET)
    inverters_zero_ac: int | None = Field(default=None, ge=0, le=FLEET)
    previous_irradiation: float | None = None

    @field_validator("previous_irradiation")
    @classmethod
    def validate_previous_irradiation(cls, value):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("previous_irradiation must be finite and non-negative")
        return value


class Reading(Conditions, Evidence):
    ac_power: float | None = None

    @field_validator("ac_power")
    @classmethod
    def validate_ac_power(cls, value):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("ac_power must be finite and non-negative")
        return value


class ReadingCreate(Conditions, Evidence):
    ac_power: float
    note: str = ""

    @field_validator("ac_power")
    @classmethod
    def validate_ac_power(cls, value):
        if not np.isfinite(value) or value < 0:
            raise ValueError("ac_power must be finite and non-negative")
        return value


class ReadingUpdate(BaseModel):
    plant_id: int | None = None
    date_time: str | None = None
    ambient_temperature: float | None = None
    module_temperature: float | None = None
    irradiation: float | None = None
    ac_power: float | None = None
    note: str | None = None
    inverters_reporting: int | None = Field(default=None, ge=0, le=FLEET)
    inverters_zero_ac: int | None = Field(default=None, ge=0, le=FLEET)
    previous_irradiation: float | None = None

    @field_validator("ambient_temperature", "module_temperature", "irradiation", "previous_irradiation", "ac_power")
    @classmethod
    def validate_optional_finite_values(cls, value):
        if value is not None and not np.isfinite(value):
            raise ValueError("Numeric inputs must be finite")
        return value

    @field_validator("irradiation", "previous_irradiation")
    @classmethod
    def validate_optional_nonnegative_irradiation(cls, value):
        if value is not None and value < 0:
            raise ValueError("Irradiation must be non-negative")
        return value

    @field_validator("ac_power")
    @classmethod
    def validate_optional_ac_power(cls, value):
        if value is not None and value < 0:
            raise ValueError("ac_power must be non-negative")
        return value


class ForecastCreate(Conditions):
    note: str = ""


class ForecastUpdate(BaseModel):
    plant_id: int | None = None
    date_time: str | None = None
    ambient_temperature: float | None = None
    module_temperature: float | None = None
    irradiation: float | None = None
    note: str | None = None

    @field_validator("ambient_temperature", "module_temperature", "irradiation")
    @classmethod
    def validate_optional_finite_weather(cls, value):
        if value is not None and not np.isfinite(value):
            raise ValueError("Weather inputs must be finite numbers")
        return value

    @field_validator("irradiation")
    @classmethod
    def validate_optional_nonnegative_irradiation(cls, value):
        if value is not None and value < 0:
            raise ValueError("irradiation must be non-negative")
        return value


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
    """Predict plant AC power; zero irradiance is a physical zero-generation case."""
    check_plant(r.plant_id)
    if r.irradiation == 0:
        return 0.0

    prediction = float(model.predict(build_row(r))[0])
    if not np.isfinite(prediction):
        logger.error("Model returned non-finite prediction for plant_id=%s", r.plant_id)
        raise HTTPException(500, "Prediction model returned an invalid value")
    return max(prediction, 0.0)


def weather_warnings(r: Conditions) -> list[str]:
    """Return non-blocking sensor plausibility warnings without altering valid inputs."""
    warnings = []
    if r.irradiation > DAY_IRR and r.module_temperature == 0:
        warnings.append(
            "Module temperature is 0°C despite measurable daytime irradiation; verify the module-temperature sensor."
        )
    return warnings


def evidence_of(obj) -> dict:
    """Read the optional evidence from a pydantic model or from a stored document."""
    get = obj.get if isinstance(obj, dict) else (lambda k: getattr(obj, k, None))
    return {k: get(k) for k in Evidence.model_fields}


def diagnose(
    r: Conditions,
    ac_power: float,
    pred: float,
    residual: float,
    center: float,
    thr: float,
    daytime: bool,
    deviation: bool | None,
    ev: dict | None = None,
) -> dict:
    ev = ev or {}
    n_rep = ev.get("inverters_reporting")
    n_zero = ev.get("inverters_zero_ac")
    prev_irr = ev.get("previous_irradiation")
    has_inverter_data = n_rep is not None or n_zero is not None

    def result(reason, causes, confidence, severity, confirm=None):
        return {
            "reason": reason,
            "likely_causes": causes,
            "confidence": confidence,
            "severity": severity,
            # Aggregate residuals cannot establish a measured count of failed inverters.
            "estimated_inverters_lost": None,
            "to_confirm": confirm,
            "evidence_used": has_inverter_data or prev_irr is not None,
        }

    if not daytime:
        return result("Night-time / below daytime-irradiance threshold; outage diagnosis not applied.",
                      [], "n/a", "none")

    # Direct inverter telemetry is independent evidence and can be assessed even
    # when residual calibration is unavailable.
    if n_rep is not None and n_zero is not None and n_zero > n_rep:
        return result(
            "Inconsistent inverter evidence: zero-output count exceeds reporting count.",
            ["Invalid inverter telemetry"], "none", "unknown",
            "Check SCADA counts and telemetry payload",
        )

    if n_rep is not None and n_rep < FLEET:
        causes = ["Missing inverter telemetry"]
        if n_zero:
            causes.append("Zero-output inverters among reporting units")
        return result(
            f"Only {n_rep} of {FLEET} inverters reported. The plant total may be understated; outage cannot be confirmed from aggregate data.",
            causes, "medium" if n_zero is not None and n_zero > 0 else "low",
            "unknown" if deviation is None else ("severe" if deviation else "moderate"),
            "Check logger communications and obtain per-inverter AC readings",
        )

    if n_zero is not None and n_zero > 0:
        if n_rep is not None and n_zero == n_rep:
            return result(
                "Every reporting inverter is at zero AC while irradiation is above the daytime threshold; possible plant outage or data-logger fault.",
                ["Possible full plant outage", "Possible data-logger fault"],
                "high", "severe", "Verify grid-trip, SCADA and inverter status logs",
            )
        return result(
            f"{n_zero} reporting inverters have zero AC output; this supports a partial inverter issue but does not quantify plant energy loss.",
            ["Zero-output inverter(s)", "Possible partial inverter outage"],
            "medium", "moderate", "Inspect affected inverter status and per-inverter power",
        )

    if deviation is None:
        return result(
            "Deviation status is undetermined because the required residual calibration is incomplete.",
            ["Calibration unavailable"], "n/a", "unknown",
            "Restore the validated residual centre and relative-rule parameters before classifying this reading",
        )
    if not deviation:
        return result("Normal: actual output is within the configured expected range.",
                      [], "n/a", "none")

    excess = abs(residual - center) / thr if thr > 0 else 0.0
    ratio = ac_power / pred if pred > 0 else None
    severity = "severe" if (excess > 2 or (ratio is not None and ratio < 0.5)) else "moderate"
    cloud = prev_irr is not None and abs(r.irradiation - prev_irr) > CLOUD_JUMP

    if residual > 0:
        return result(
            "Actual output is above model expectation; check irradiance timing, sensor lag and cloud-edge effects.",
            ["Irradiance sensor timing mismatch", "Cloud-edge effect"],
            "low", severity, "Compare adjacent weather and power readings",
        )

    causes = ["Plant-wide derating or curtailment", "Soiling or shading", "Irradiance sensor over-reading"]
    if cloud:
        causes.insert(0, "Rapid irradiance change")
    return result(
        f"Actual output is {abs(residual):.1f} kW below the model expectation. Aggregate evidence cannot isolate the physical cause.",
        causes, "low", severity,
        "Compare inverter-level outputs, grid curtailment logs and weather sensor readings",
    )


def assess(r: Conditions, ac_power: float, evidence: dict | None = None) -> dict:
    """Compare actual AC power to model expectation and explain the residual."""
    pid = str(r.plant_id)
    pred = predict_kw(r)
    residual = float(ac_power) - pred
    thr = THRESHOLDS[pid]
    center = CENTERS.get(pid)

    daytime = r.irradiation > DAY_IRR
    if center is None:
        size_rule = None
    else:
        size_rule = bool(daytime and abs(residual - center) > thr)

    if REL is None:
        relative_rule = None
    else:
        relative_rule = bool(
            daytime
            and pred > float(REL["min_predicted_kw"])
            and ac_power < float(REL["ratio"]) * pred
        )

    # A positive available check detects deviation. A negative result is
    # conclusive only when both required calibration checks were evaluated.
    available_rules = [rule for rule in (size_rule, relative_rule) if rule is not None]
    if any(rule is True for rule in available_rules):
        deviation = True
    elif size_rule is not None and relative_rule is not None:
        deviation = False
    else:
        deviation = None

    calibration_status = (
        "complete" if size_rule is not None and relative_rule is not None
        else "partial" if size_rule is not None or relative_rule is not None
        else "incomplete"
    )

    center_for_display = center
    diagnosis = diagnose(
        r, ac_power, pred, residual, center or 0.0, thr, daytime,
        deviation, evidence,
    )
    return {
        "plant_id": r.plant_id,
        "actual_ac_power_kw": round(float(ac_power), 1),
        "predicted_ac_power_kw": round(pred, 1),
        "residual_kw": round(residual, 1),
        "residual_centre_kw": round(center_for_display, 1) if center_for_display is not None else None,
        "threshold_kw": round(thr, 1),
        "daytime": daytime,
        "size_rule": size_rule,
        "relative_rule": relative_rule,
        "deviation": deviation,
        "calibration_status": calibration_status,
        "input_warnings": weather_warnings(r),
        "calculation": (
            f"residual = actual - predicted = {ac_power:.1f} - {pred:.1f} = {residual:.1f} kW; "
            + (f"allowed band = {center:.1f} +/- {thr:.1f} kW"
               if center is not None else "residual centre is not configured")
        ),
        **diagnosis,
    }


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
    return {
        "plant_id": r.plant_id,
        "predicted_ac_power_kw": round(predict_kw(r), 1),
        "input_warnings": weather_warnings(r),
    }


@app.post("/deviation")
def deviation(r: Reading):
    if r.ac_power is None:
        raise HTTPException(422, "ac_power is required for /deviation")
    return assess(r, r.ac_power, evidence_of(r))


# ------------------------------------------------------------------ CRUD: readings (Plant Engineer)
@app.post("/readings", status_code=201)                      # CREATE
def create_reading(r: ReadingCreate, _user: dict = Depends(require_role('operator'))):
    doc = r.model_dump()
    doc.update(assess(r, r.ac_power, evidence_of(r)))
    doc["created_at"] = doc["updated_at"] = now()
    doc["_id"] = db.readings.insert_one(doc).inserted_id
    return out(doc)


@app.get("/readings")                                        # READ (list, with filters)
def list_readings(plant_id: int | None = None, deviation: bool | None = None,
                  limit: int = Query(50, ge=1, le=200), skip: int = Query(0, ge=0),
                  _user: dict = Depends(require_role("viewer"))):
    query = {}
    if plant_id is not None:
        query["plant_id"] = plant_id
    if deviation is not None:
        query["deviation"] = deviation
    cursor = db.readings.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return {"total": db.readings.count_documents(query), "items": [out(d) for d in cursor]}


@app.get("/readings/{reading_id}")                           # READ (one)
def get_reading(reading_id: str, _user: dict = Depends(require_role('viewer'))):
    return out(fetch_or_404(db.readings, reading_id))


@app.put("/readings/{reading_id}")                           # UPDATE
def update_reading(reading_id: str, changes: ReadingUpdate, _user: dict = Depends(require_role('admin'))):
    doc = fetch_or_404(db.readings, reading_id)
    new = {**doc, **changes.model_dump(exclude_unset=True)}
    if new.get("ac_power") is not None:
        new.update(assess(conditions_of(new), new["ac_power"], evidence_of(new)))
    new["updated_at"] = now()
    new.pop("_id")
    db.readings.update_one({"_id": doc["_id"]}, {"$set": new})
    return out({**new, "_id": doc["_id"]})


@app.delete("/readings/{reading_id}")                        # DELETE
def delete_reading(reading_id: str, _user: dict = Depends(require_role('admin'))):
    if db.readings.delete_one({"_id": to_oid(reading_id)}).deleted_count == 0:
        raise HTTPException(404, "Not found")
    return {"deleted": reading_id}


# ------------------------------------------------------------------ CRUD: forecasts (DISCOM)
@app.post("/forecasts", status_code=201)                     # CREATE
def create_forecast(f: ForecastCreate, _user: dict = Depends(require_role('operator'))):
    doc = f.model_dump()
    doc["predicted_ac_power_kw"] = round(predict_kw(f), 1)
    doc["created_at"] = doc["updated_at"] = now()
    doc["_id"] = db.forecasts.insert_one(doc).inserted_id
    return out(doc)


@app.get("/forecasts")                                       # READ (list)
def list_forecasts(plant_id: int | None = None,
                   limit: int = Query(50, ge=1, le=200), skip: int = Query(0, ge=0),
                  _user: dict = Depends(require_role("viewer"))):
    query = {} if plant_id is None else {"plant_id": plant_id}
    cursor = db.forecasts.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return {"total": db.forecasts.count_documents(query), "items": [out(d) for d in cursor]}


@app.get("/forecasts/{forecast_id}")                         # READ (one)
def get_forecast(forecast_id: str, _user: dict = Depends(require_role('viewer'))):
    return out(fetch_or_404(db.forecasts, forecast_id))


@app.put("/forecasts/{forecast_id}")                         # UPDATE
def update_forecast(forecast_id: str, changes: ForecastUpdate, _user: dict = Depends(require_role('admin'))):
    doc = fetch_or_404(db.forecasts, forecast_id)
    new = {**doc, **changes.model_dump(exclude_unset=True)}
    new["predicted_ac_power_kw"] = round(predict_kw(conditions_of(new)), 1)
    new["updated_at"] = now()
    new.pop("_id")
    db.forecasts.update_one({"_id": doc["_id"]}, {"$set": new})
    return out({**new, "_id": doc["_id"]})


@app.delete("/forecasts/{forecast_id}")                      # DELETE
def delete_forecast(forecast_id: str, _user: dict = Depends(require_role('admin'))):
    if db.forecasts.delete_one({"_id": to_oid(forecast_id)}).deleted_count == 0:
        raise HTTPException(404, "Not found")
    return {"deleted": forecast_id}

