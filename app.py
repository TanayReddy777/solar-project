import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model" if (BASE / "model").exists() else BASE
model = joblib.load(MODEL_DIR / "solar_model.joblib")
config = json.loads((MODEL_DIR / "model_config.json").read_text())

FEATURES = config["features"]
THRESHOLDS = {str(k): float(v) for k, v in config["deviation_thresholds_kw"].items()}
PLANT_2_ID = int(config["plant_2_id"])        # plant_code = 1 for this plant, 0 for the other
DAY_IRR = float(config["daytime_irradiation"])

app = FastAPI(title="Solar power API", version="1.0")


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


class Reading(BaseModel):
    plant_id: int
    date_time: str
    ambient_temperature: float
    module_temperature: float
    irradiation: float
    ac_power: float | None = None      # needed only for /deviation


def build_row(r: Reading) -> pd.DataFrame:
    if str(r.plant_id) not in THRESHOLDS:
        raise HTTPException(422, f"Unknown plant_id {r.plant_id}. Use one of {list(THRESHOLDS)}")
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


@app.get("/")
def health():
    return {"status": "ok", "endpoints": ["/predict", "/deviation", "/docs"]}


@app.post("/predict")
def predict(r: Reading):
    pred = max(float(model.predict(build_row(r))[0]), 0.0)
    return {"plant_id": r.plant_id, "predicted_ac_power_kw": round(pred, 1)}


@app.post("/deviation")
def deviation(r: Reading):
    if r.ac_power is None:
        raise HTTPException(422, "ac_power is required for /deviation")
    if r.irradiation <= DAY_IRR:
        return {"deviation": False, "reason": "night-time reading, not checked"}
    pred = max(float(model.predict(build_row(r))[0]), 0.0)
    residual = r.ac_power - pred
    thr = THRESHOLDS[str(r.plant_id)]
    return {
        "plant_id": r.plant_id,
        "actual_ac_power_kw": r.ac_power,
        "predicted_ac_power_kw": round(pred, 1),
        "residual_kw": round(residual, 1),
        "threshold_kw": round(thr, 1),
        "deviation": bool(abs(residual) > thr),
    }
