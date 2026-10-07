from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os

app = FastAPI(title="Kosi Flood Risk Predictor")

from fastapi.staticfiles import StaticFiles
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="dashboard")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "../models"
HORIZONS = ["24h", "3day", "5day"]
STATIONS = ["Baltara", "Basua", "Birpur", "Jainagar", "Kursela"]

models = {}
columns = {}
feature_sets = {}
for h in HORIZONS:
    models[h] = joblib.load(os.path.join(MODEL_DIR, f"model_{h}.pkl"))
    columns[h] = joblib.load(os.path.join(MODEL_DIR, f"columns_{h}.pkl"))
    feature_sets[h] = joblib.load(os.path.join(MODEL_DIR, f"features_{h}.pkl"))

# resampled_data already carries Danger/Warning/HFL thresholds per row, merged back in notebook 03
resampled_data = pd.read_parquet("../data/processed/resampled_stations.parquet")

# per-station thresholds, looked up by name — used by /predict since it doesn't get a full history row otherwise
thresholds_lookup = (
    resampled_data.sort_values("time")
    .groupby("station_clean")[["Danger Level (m)", "Warning Level (m)", "HFL / Highest Flood Level (m)"]]
    .last()
    .to_dict("index")
)


def compute_all_features(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Compute every feature any horizon's model might need, on the latest row."""
    df = df.sort_values("time").reset_index(drop=True)

    df["wse_change_1step"] = df["WSE"].diff()
    df["wse_change_4step"] = df["WSE"].diff(4)
    df["wse_change_20step"] = df["WSE"].diff(20)
    df["wse_rolling_std_4"] = df["WSE"].rolling(4, min_periods=1).std().fillna(0)
    df["wse_rolling_std_20"] = df["WSE"].rolling(20, min_periods=1).std().fillna(0)

    latest = df.iloc[[-1]].copy()
    latest["station_clean"] = station

    station_thresholds = thresholds_lookup[station]
    latest["dist_to_warning_boundary"] = station_thresholds["Warning Level (m)"] - latest["WSE"]
    latest["dist_to_danger_boundary"] = station_thresholds["Danger Level (m)"] - latest["WSE"]

    return latest


def predict_for_horizon(latest: pd.DataFrame, horizon: str):
    cols_needed = feature_sets[horizon]
    X = pd.get_dummies(latest[cols_needed + ["station_clean"]], columns=["station_clean"])
    X = X.reindex(columns=columns[horizon], fill_value=0)

    model = models[horizon]
    pred = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    return pred, proba


class Reading(BaseModel):
    time: str   # ISO format, e.g. "2026-09-20T06:00:00"
    wse: float

class PredictRequest(BaseModel):
    station: str
    readings: List[Reading]   # most recent 20 readings, oldest first
    horizon: str = "24h"      # "24h", "3day", or "5day"


@app.post("/predict")
def predict(req: PredictRequest):
    if req.station not in STATIONS:
        raise HTTPException(400, f"Unknown station. Must be one of {STATIONS}")
    if req.horizon not in HORIZONS:
        raise HTTPException(400, f"Unknown horizon. Must be one of {HORIZONS}")
    if len(req.readings) < 20:
        raise HTTPException(400, "Need at least 20 recent readings to compute features")

    df = pd.DataFrame([{"time": r.time, "WSE": r.wse} for r in req.readings])
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)

    latest = compute_all_features(df, req.station)
    pred, proba = predict_for_horizon(latest, req.horizon)

    return {
        "station": req.station,
        "horizon": req.horizon,
        "predicted_risk": pred,
        "probabilities": {k: round(float(v), 3) for k, v in proba.items()}
    }


@app.get("/stations/current")
def stations_current():
    results = []
    for station in STATIONS:
        station_df = resampled_data[resampled_data["station_clean"] == station].sort_values("time")
        if len(station_df) < 20:
            continue
        recent = station_df.tail(20)[["time", "WSE"]].copy()

        latest = compute_all_features(recent, station)

        station_result = {
            "station": station,
            "latest_wse": round(float(latest["WSE"].values[0]), 2),
            "latest_time": str(latest["time"].values[0]),
            "forecasts": {}
        }

        for h in HORIZONS:
            pred, proba = predict_for_horizon(latest, h)
            station_result["forecasts"][h] = {
                "predicted_risk": pred,
                "confidence": round(float(max(proba.values())), 3)
            }

        results.append(station_result)

    return {"stations": results}


@app.get("/stations/{station}/history")
def station_history(station: str):
    if station not in STATIONS:
        raise HTTPException(404, "Unknown station")

    station_df = resampled_data[resampled_data["station_clean"] == station].sort_values("time").tail(20)
    thresholds = station_df.iloc[-1]

    return {
        "station": station,
        "readings": [
            {"time": str(t), "wse": round(float(w), 2)}
            for t, w in zip(station_df["time"], station_df["WSE"])
        ],
        "danger_level": round(float(thresholds["Danger Level (m)"]), 2),
        "warning_level": round(float(thresholds["Warning Level (m)"]), 2),
        "hfl": round(float(thresholds["HFL / Highest Flood Level (m)"]), 2),
    }


@app.get("/")
def root():
    return {"status": "Kosi Flood Predictor API is running"}