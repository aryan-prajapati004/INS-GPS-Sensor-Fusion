"""
INS/GPS Fusion — inference API.

Serves 12 trained deep-learning sensor-fusion models for CPU inference on
Render. Model artifacts (.pkl files containing state_dict + scalers +
metrics) are expected in the ARTIFACT_DIR directory, one per model:
    artifacts/RNN.pkl, artifacts/LSTM.pkl, ... artifacts/BiMamba.pkl

Each .pkl is a pickle of a dict with keys:
    model_name, state_dict, history, f_scaler, t_scaler,
    feature_cols, window, stride, metrics, train_time_s
(this matches exactly what the training notebook exports)
"""
import csv
import math
import os
import pickle
import time
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import MODEL_NAMES, MODEL_PALETTE, N_FEAT, STRIDE, WINDOW, make_model

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

app = FastAPI(
    title="INS/GPS Fusion Inference API",
    description="CPU inference for 12 deep-learning sensor-fusion models "
                 "(RNN, LSTM, GRU, BiRNN, BiLSTM, Transformer, BERT, TFT, "
                 "S4, Mamba, Mamba2, BiMamba).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your dashboard's origin once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE: Dict[str, dict] = {}  # model_name -> {"model": nn.Module, "artifact": dict}


def _infer_hyperparams(model_name: str, state_dict: dict) -> dict:
    """Infer constructor kwargs from a saved state_dict so that the model
    architecture matches the trained checkpoint exactly."""
    sd = state_dict
    kwargs = {}

    # ── Recurrent models (RNN, LSTM, GRU, BiRNN, BiLSTM) ───────────────
    rnn_key_map = {
        'RNN': 'rnn', 'BiRNN': 'rnn',
        'LSTM': 'lstm', 'BiLSTM': 'lstm',
        'GRU': 'gru',
    }
    if model_name in rnn_key_map:
        rnn_attr = rnn_key_map[model_name]
        # hidden = size of bias_ih_l0 (for LSTM it's 4*hidden, for GRU 3*hidden)
        bias_key = f'{rnn_attr}.bias_ih_l0'
        raw_h = sd[bias_key].shape[0]
        if model_name in ('LSTM', 'BiLSTM'):
            kwargs['hidden'] = raw_h // 4
        elif model_name == 'GRU':
            kwargs['hidden'] = raw_h // 3
        else:
            kwargs['hidden'] = raw_h
        # n_layers: count distinct weight_ih_l* keys (excluding _reverse)
        layer_keys = [k for k in sd if k.startswith(f'{rnn_attr}.weight_ih_l')
                      and '_reverse' not in k]
        kwargs['layers'] = len(layer_keys)
        return kwargs

    # ── Transformer ────────────────────────────────────────────────────
    if model_name == 'Transformer':
        kwargs['d_model'] = sd['proj.bias'].shape[0]
        # Count encoder layers
        layer_indices = {int(k.split('.')[2]) for k in sd
                         if k.startswith('encoder.layers.')}
        kwargs['n_layers'] = len(layer_indices)
        # n_heads: in_proj_weight first dim = 3 * d_model, out_proj = d_model
        # dim_feedforward: linear1.weight first dim
        if 'encoder.layers.0.linear1.weight' in sd:
            kwargs['dim_ff'] = sd['encoder.layers.0.linear1.weight'].shape[0]
        return kwargs

    # ── BERT ───────────────────────────────────────────────────────────
    if model_name == 'BERT':
        kwargs['d_model'] = sd['embedding.weight'].shape[0]
        layer_indices = {int(k.split('.')[2]) for k in sd
                         if k.startswith('transformer_encoder.layers.')}
        kwargs['n_layers'] = len(layer_indices)
        if 'transformer_encoder.layers.0.linear1.weight' in sd:
            kwargs['dim_ff'] = sd['transformer_encoder.layers.0.linear1.weight'].shape[0]
        return kwargs

    # ── TFT ────────────────────────────────────────────────────────────
    if model_name == 'TFT':
        kwargs['hidden'] = sd['input_grn.fc1.bias'].shape[0]
        # n_heads: attn.in_proj_weight first dim / (3 * hidden)
        h = kwargs['hidden']
        if 'attn.in_proj_weight' in sd:
            kwargs['n_heads'] = sd['attn.in_proj_weight'].shape[0] // (3 * h)
        return kwargs

    # ── S4 ─────────────────────────────────────────────────────────────
    if model_name == 'S4':
        kwargs['d_model'] = sd['proj.bias'].shape[0]
        kwargs['d_state'] = sd['blocks.0.kernel.A_real'].shape[1]
        layer_indices = {int(k.split('.')[1]) for k in sd
                         if k.startswith('blocks.')}
        kwargs['n_layers'] = len(layer_indices)
        return kwargs

    # ── Mamba / BiMamba ────────────────────────────────────────────────
    if model_name in ('Mamba', 'BiMamba'):
        kwargs['d_model'] = sd['proj.bias'].shape[0]
        # d_state from SSM A_log second dim
        prefix = 'blocks.0' if model_name == 'Mamba' else 'fwd_blocks.0'
        kwargs['d_state'] = sd[f'{prefix}.ssm.A_log'].shape[1]
        block_prefix = 'blocks.' if model_name == 'Mamba' else 'fwd_blocks.'
        layer_indices = {int(k.split('.')[1]) for k in sd
                         if k.startswith(block_prefix)}
        kwargs['n_layers'] = len(layer_indices)
        return kwargs

    # ── Mamba2 ─────────────────────────────────────────────────────────
    if model_name == 'Mamba2':
        kwargs['d_model'] = sd['proj.bias'].shape[0]
        # d_state: BC_proj output = 2 * d_state, so d_state = shape[0] / 2
        # But BC_proj takes head_dim input and outputs 2*d_state
        kwargs['d_state'] = sd['blocks.0.BC_proj.weight'].shape[0] // 2  # approximate
        # But need exact: BC_proj.weight shape = [2*d_state, head_dim]
        # Also infer n_heads from A_log
        kwargs['n_heads'] = sd['blocks.0.A_log'].shape[0]
        layer_indices = {int(k.split('.')[1]) for k in sd
                         if k.startswith('blocks.')}
        kwargs['n_layers'] = len(layer_indices)
        return kwargs

    return kwargs


def _load_artifact(model_name: str) -> dict:
    """Load a .pkl artifact from disk and build a ready-to-eval model."""
    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'. "
                                                      f"Choose from {MODEL_NAMES}")
    if model_name in _CACHE:
        return _CACHE[model_name]

    path = os.path.join(ARTIFACT_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise HTTPException(status_code=503,
                             detail=f"Artifact missing for '{model_name}' at {path}. "
                                    f"Did you upload the trained .pkl files?")
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    n_feat = len(artifact.get("feature_cols", [])) or N_FEAT
    hp = _infer_hyperparams(model_name, artifact["state_dict"])
    model = make_model(model_name, n_feat, **hp)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    entry = {"model": model, "artifact": artifact}
    _CACHE[model_name] = entry
    return entry


@app.on_event("startup")
def preload_models():
    """Best-effort preload so first requests aren't slow. Missing files are
    skipped silently — /models will report which ones actually loaded."""
    for name in MODEL_NAMES:
        try:
            _load_artifact(name)
        except HTTPException:
            pass


# ── Schemas ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    window: List[List[float]] = Field(
        ..., description=f"A [{WINDOW}, {N_FEAT}] sensor window: {WINDOW} timesteps "
                          f"x {N_FEAT} features (yaw_rate, accel_lon, accel_lat, dt, "
                          f"4x wheel speeds, vehicle speed, steering angle, heading_sin, "
                          f"heading_cos, gps_velocity)."
    )


class PredictResponse(BaseModel):
    model_name: str
    v_N: float
    v_E: float
    inference_ms: float


class TrajectoryRequest(BaseModel):
    windows: List[List[List[float]]] = Field(
        ..., description="A list of sensor windows, one per timestep to predict, "
                          "each shaped [window, n_feat]."
    )
    dt: List[float] = Field(..., description="Per-step dt (seconds) matching `windows`.")
    start_PN: float = 0.0
    start_PE: float = 0.0


class TrajectoryResponse(BaseModel):
    model_name: str
    pred_PN: List[float]
    pred_PE: List[float]
    pred_v_N: List[float]
    pred_v_E: List[float]
    inference_ms: float


class ModelInfo(BaseModel):
    model_name: str
    color: str
    loaded: bool
    param_count: Optional[int] = None
    metrics: Optional[dict] = None
    train_time_s: Optional[float] = None


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(_CACHE.keys())}


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    out = []
    for name in MODEL_NAMES:
        try:
            entry = _load_artifact(name)
            model, artifact = entry["model"], entry["artifact"]
            out.append(ModelInfo(
                model_name=name,
                color=MODEL_PALETTE.get(name, "#888888"),
                loaded=True,
                param_count=sum(p.numel() for p in model.parameters()),
                metrics=artifact.get("metrics"),
                train_time_s=artifact.get("train_time_s"),
            ))
        except HTTPException:
            out.append(ModelInfo(model_name=name, color=MODEL_PALETTE.get(name, "#888888"),
                                  loaded=False))
    return out


@app.get("/leaderboard")
def leaderboard():
    """All loaded models sorted by ADE (m), ascending — lower is better."""
    rows = []
    for name in MODEL_NAMES:
        try:
            entry = _load_artifact(name)
            m = entry["artifact"].get("metrics", {})
            rows.append({"model_name": name, "color": MODEL_PALETTE.get(name), **m})
        except HTTPException:
            continue
    rows.sort(key=lambda r: r.get("ADE (m)", float("inf")))
    return {"leaderboard": rows}


@app.post("/predict/{model_name}", response_model=PredictResponse)
def predict(model_name: str, req: PredictRequest):
    entry = _load_artifact(model_name)
    model, artifact = entry["model"], entry["artifact"]
    f_scaler, t_scaler = artifact["f_scaler"], artifact["t_scaler"]

    x = np.asarray(req.window, dtype=np.float32)
    expected_window = artifact.get("window", WINDOW)
    if x.shape != (expected_window, len(artifact.get("feature_cols", range(N_FEAT)))):
        raise HTTPException(
            status_code=422,
            detail=f"Expected window shape ({expected_window}, "
                   f"{len(artifact.get('feature_cols', []))}), got {x.shape}"
        )

    x_scaled = f_scaler.transform(x)
    x_t = torch.from_numpy(x_scaled).float().unsqueeze(0)  # [1, W, F]

    t0 = time.perf_counter()
    with torch.no_grad():
        pred_scaled = model(x_t).numpy()
    inference_ms = (time.perf_counter() - t0) * 1000

    pred = t_scaler.inverse_transform(pred_scaled)[0]
    return PredictResponse(model_name=model_name, v_N=float(pred[0]), v_E=float(pred[1]),
                            inference_ms=round(inference_ms, 3))


@app.post("/predict_trajectory/{model_name}", response_model=TrajectoryResponse)
def predict_trajectory(model_name: str, req: TrajectoryRequest):
    entry = _load_artifact(model_name)
    model, artifact = entry["model"], entry["artifact"]
    f_scaler, t_scaler = artifact["f_scaler"], artifact["t_scaler"]
    stride = artifact.get("stride", STRIDE)

    if len(req.windows) != len(req.dt):
        raise HTTPException(status_code=422, detail="`windows` and `dt` must be the same length")
    if len(req.windows) == 0:
        raise HTTPException(status_code=422, detail="`windows` must be non-empty")

    t0 = time.perf_counter()
    batch = np.stack([f_scaler.transform(np.asarray(w, dtype=np.float32)) for w in req.windows])
    x_t = torch.from_numpy(batch).float()
    with torch.no_grad():
        pred_scaled = model(x_t).numpy()
    inference_ms = (time.perf_counter() - t0) * 1000

    pred_vel = t_scaler.inverse_transform(pred_scaled)  # [T, 2] -> (v_N, v_E)

    n = len(req.dt)
    pred_PN = np.zeros(n)
    pred_PE = np.zeros(n)
    pred_PN[0] = req.start_PN
    pred_PE[0] = req.start_PE
    for t in range(1, n):
        step_dt = stride * req.dt[t]
        pred_PN[t] = pred_PN[t - 1] + pred_vel[t, 0] * step_dt
        pred_PE[t] = pred_PE[t - 1] + pred_vel[t, 1] * step_dt

    return TrajectoryResponse(
        model_name=model_name,
        pred_PN=pred_PN.tolist(), pred_PE=pred_PE.tolist(),
        pred_v_N=pred_vel[:, 0].tolist(), pred_v_E=pred_vel[:, 1].tolist(),
        inference_ms=round(inference_ms, 3),
    )

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

RESET_INTERVAL = 200  # Reset predicted position to GT every N windows to prevent drift

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

REQUIRED_SENSOR_COLS = [
    "Yaw Rate (deg/sec)",
    "Indicated Longitudinal Acceleration (g)",
    "Indicated Lateral Acceleration (g)",
    "Sample period (seconds)",
    "Wheel Speed Front Left (rad/sec)",
    "Wheel Speed Front Right (rad/sec)",
    "Wheel Speed Rear Left (rad/sec)",
    "Wheel Speed Rear Right (rad/sec)",
    "Indicated Vehicle Speed (km/hr)",
    "Steering Angle (degrees)",
    "Heading (degrees)",
    "Velocity (km/hr)",
]

GT_COL_SETS = [
    ["PN", "PE"],
    ["Latitude (degrees)", "Longitude (degrees)"],
]


def _load_dataset(filename: str):
    """Load a dataset CSV and return (raw_features, gt_PN, gt_PE, dt_arr).
    Handles both formats: Aligned_Data_preprocessed.csv (has PN/PE columns)
    and v1.csv (Lat/Lon -> PN/PE via local tangent plane projection)."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw = []
    gt_PN = []
    gt_PE = []
    dt_arr = []

    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        has_PN = 'PN' in reader.fieldnames
        lat_ref = lon_ref = None

        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                yaw_rate = float(row.get("Yaw Rate (deg/sec)", 0))
                accel_lon = float(row.get("Indicated Longitudinal Acceleration (g)", 0))
                accel_lat = float(row.get("Indicated Lateral Acceleration (g)", 0))
                dt = float(row.get("Sample period (seconds)", 0.1))
                wfl = float(row.get("Wheel Speed Front Left (rad/sec)", 0))
                wfr = float(row.get("Wheel Speed Front Right (rad/sec)", 0))
                wrl = float(row.get("Wheel Speed Rear Left (rad/sec)", 0))
                wrr = float(row.get("Wheel Speed Rear Right (rad/sec)", 0))
                veh_speed = float(row.get("Indicated Vehicle Speed (km/hr)", 0))
                steer = float(row.get("Steering Angle (degrees)", 0))
                heading = float(row.get("Heading (degrees)", 0))
                heading_sin = math.sin(math.radians(heading))
                heading_cos = math.cos(math.radians(heading))
                gps_v = float(row.get("Velocity (km/hr)", 0))

                raw.append([yaw_rate, accel_lon, accel_lat, dt, wfl, wfr, wrl, wrr,
                            veh_speed, steer, heading_sin, heading_cos, gps_v])
                dt_arr.append(dt)

                if has_PN:
                    gt_PN.append(float(row.get("PN", 0)))
                    gt_PE.append(float(row.get("PE", 0)))
                else:
                    lat = float(row.get("Latitude (degrees)", 0))
                    lon = float(row.get("Longitude (degrees)", 0))
                    if lat_ref is None:
                        lat_ref, lon_ref = lat, lon
                    pn = (lat - lat_ref) * 111320.0
                    pe = (lon - lon_ref) * 111320.0 * math.cos(math.radians(lat_ref))
                    gt_PN.append(pn)
                    gt_PE.append(pe)
            except (ValueError, TypeError):
                continue

    return raw, gt_PN, gt_PE, dt_arr


def _validate_and_clean_csv(filepath: str) -> dict:
    import io
    import csv
    import math
    import numpy as np
    
    report = {
        "filename": "",
        "valid": False,
        "total_rows": 0,
        "valid_rows": 0,
        "columns_found": [],
        "sensor_cols_present": [],
        "sensor_cols_missing": [],
        "gt_format": "none",
        "issues": [],
        "cleaning_applied": [],
        "column_stats": {},
        "min_window_met": False,
        "usable_windows": 0,
    }

    if not os.path.exists(filepath):
        report["issues"].append({"severity": "error", "message": "File not found."})
        return report

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except Exception as e:
        report["issues"].append({"severity": "error", "message": f"File read error: {str(e)}"})
        return report

    try:
        f_io = io.StringIO(content)
        reader = csv.DictReader(f_io)
        if not reader.fieldnames:
            report["issues"].append({"severity": "error", "message": "No columns found."})
            return report
        columns_found = [n.strip() for n in reader.fieldnames]
        report["columns_found"] = columns_found
    except Exception as e:
        report["issues"].append({"severity": "error", "message": f"CSV parse error: {str(e)}"})
        return report

    present = []
    missing = []
    for col in REQUIRED_SENSOR_COLS:
        if col in columns_found:
            present.append(col)
        else:
            missing.append(col)
    
    report["sensor_cols_present"] = present
    report["sensor_cols_missing"] = missing

    if missing:
        report["issues"].append({"severity": "error", "message": f"Missing required columns: {missing}"})

    gt_format = "none"
    if "PN" in columns_found and "PE" in columns_found:
        gt_format = "PN/PE"
    elif "Latitude (degrees)" in columns_found and "Longitude (degrees)" in columns_found:
        gt_format = "Lat/Lon"
    else:
        report["issues"].append({"severity": "warning", "message": "No recognized ground-truth columns found (PN/PE or Lat/Lon)."})
    report["gt_format"] = gt_format

    f_io.seek(0)
    reader = csv.DictReader(f_io)
    reader.fieldnames = columns_found

    rows = []
    for row in reader:
        if all(v is None or str(v).strip() == "" for v in row.values()):
            continue
        cleaned_row = {k: (str(v).strip() if v is not None else "") for k, v in row.items()}
        rows.append(cleaned_row)

    total_rows = len(rows)
    report["total_rows"] = total_rows

    if total_rows == 0:
        report["issues"].append({"severity": "error", "message": "CSV is empty."})
        return report

    col_non_numeric = {col: 0 for col in present}
    col_values = {col: [None]*total_rows for col in present}

    for i, row in enumerate(rows):
        for col in present:
            val = row.get(col, "")
            if val == "":
                col_non_numeric[col] += 1
            else:
                try:
                    v = float(val)
                    if math.isnan(v):
                        col_non_numeric[col] += 1
                    else:
                        col_values[col][i] = v
                except ValueError:
                    col_non_numeric[col] += 1

    for col in present:
        missing_count = col_non_numeric[col]
        if missing_count > 0:
            missing_pct = missing_count / total_rows
            if missing_pct < 0.20:
                report["cleaning_applied"].append(f"Interpolated {missing_count} missing values in {col}")
                vals = col_values[col]
                valid_indices = [i for i, v in enumerate(vals) if v is not None]
                if valid_indices:
                    valid_vals = [vals[i] for i in valid_indices]
                    interp_vals = np.interp(range(total_rows), valid_indices, valid_vals)
                    for i in range(total_rows):
                        col_values[col][i] = interp_vals[i]
                        rows[i][col] = str(interp_vals[i])
                    col_non_numeric[col] = 0
            else:
                report["issues"].append({"severity": "warning", "message": f"Column {col} has >=20% missing values ({missing_pct*100:.1f}%), leaving as-is."})

    valid_rows = 0
    for i in range(total_rows):
        is_valid = True
        for col in present:
            if col_values[col][i] is None:
                is_valid = False
                break
        if is_valid:
            valid_rows += 1
    report["valid_rows"] = valid_rows

    report["min_window_met"] = valid_rows >= WINDOW
    if not report["min_window_met"]:
        report["issues"].append({"severity": "error", "message": f"Total valid rows ({valid_rows}) < WINDOW ({WINDOW})"})
    
    report["usable_windows"] = max(0, (valid_rows - WINDOW) // STRIDE + 1)
    
    if len(missing) == 0 and report["min_window_met"]:
        report["valid"] = True

    key_cols = ["Yaw Rate (deg/sec)", "Indicated Longitudinal Acceleration (g)", "Indicated Lateral Acceleration (g)", "Indicated Vehicle Speed (km/hr)"]
    for col in present:
        valid_vals = [v for v in col_values[col] if v is not None]
        stats = {"non_numeric": col_non_numeric[col], "outliers": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
        if valid_vals:
            stats["min"] = float(np.min(valid_vals))
            stats["max"] = float(np.max(valid_vals))
            stats["mean"] = float(np.mean(valid_vals))
            
            if col in key_cols and len(valid_vals) >= 4:
                q1 = np.percentile(valid_vals, 25)
                q3 = np.percentile(valid_vals, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                stats["outliers"] = int(sum(1 for v in valid_vals if v < lower or v > upper))
        
        report["column_stats"][col] = stats

    if "Sample period (seconds)" in present:
        dupes = 0
        cum_time = 0.0
        seen_times = set()
        for i in range(total_rows):
            dt_val = col_values["Sample period (seconds)"][i]
            if dt_val is not None:
                cum_time += dt_val
                cum_round = round(cum_time, 3)
                if cum_round in seen_times:
                    dupes += 1
                seen_times.add(cum_round)
        if dupes > 0:
            report["issues"].append({"severity": "warning", "message": f"Found {dupes} possible duplicate timestamps based on cumulative Sample period."})

    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns_found)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        report["issues"].append({"severity": "error", "message": f"Failed to save cleaned CSV: {str(e)}"})

    return report


# -- Simulate schemas --------------------------------------------------------
class SimulateRequest(BaseModel):
    model_names: List[str]
    dataset: str = "v1.csv"


class SimulateResult(BaseModel):
    model_name: str
    color: str
    pred_PN: List[float]
    pred_PE: List[float]
    errors: List[float]
    ade: float
    fde: float
    inference_ms: float


class SimulateResponse(BaseModel):
    gt_PN: List[float]
    gt_PE: List[float]
    total_windows: int
    results: List[SimulateResult]


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """Run full simulation: load dataset, build windows, infer, integrate
    with periodic GT reset to prevent drift, compute errors."""
    raw, gt_PN_all, gt_PE_all, dt_all = _load_dataset(req.dataset)

    # DOWNSAMPLE for Render Free Tier: Limit to first 2500 rows
    # This ensures the 0.1 CPU core can process the request in ~5 seconds.
    MAX_ROWS = 2500
    if len(raw) > MAX_ROWS:
        raw = raw[:MAX_ROWS]
        gt_PN_all = gt_PN_all[:MAX_ROWS]
        gt_PE_all = gt_PE_all[:MAX_ROWS]
        dt_all = dt_all[:MAX_ROWS]

    # Build sliding windows
    windows: List[list] = []
    gt_pn: List[float] = []
    gt_pe: List[float] = []
    dt_win: List[float] = []
    for i in range(0, len(raw) - WINDOW + 1, STRIDE):
        windows.append(raw[i:i + WINDOW])
        idx = i + WINDOW - 1
        gt_pn.append(gt_PN_all[idx])
        gt_pe.append(gt_PE_all[idx])
        dt_win.append(dt_all[idx])

    if not windows:
        raise HTTPException(status_code=422, detail="Dataset too short for windowing")

    n = len(windows)
    gt_pn_np = np.array(gt_pn)
    gt_pe_np = np.array(gt_pe)
    results = []

    for model_name in req.model_names:
        entry = _load_artifact(model_name)
        model_nn, artifact = entry["model"], entry["artifact"]
        f_scaler, t_scaler = artifact["f_scaler"], artifact["t_scaler"]
        stride_val = artifact.get("stride", STRIDE)

        t0 = time.perf_counter()

        # Batched inference (avoids OOM on large datasets)
        all_pred = []
        batch_sz = 512
        for bi in range(0, n, batch_sz):
            batch_raw = windows[bi:bi + batch_sz]
            batch = np.stack([f_scaler.transform(np.asarray(w, dtype=np.float32))
                              for w in batch_raw])
            x_t = torch.from_numpy(batch).float()
            with torch.no_grad():
                pred_scaled = model_nn(x_t).numpy()
            all_pred.append(t_scaler.inverse_transform(pred_scaled))

        pred_vel = np.concatenate(all_pred, axis=0)  # [n, 2]
        inference_ms = (time.perf_counter() - t0) * 1000

        # Pure integration (no reset to GT) to see raw accumulated drift
        pred_PN_arr = np.zeros(n)
        pred_PE_arr = np.zeros(n)
        pred_PN_arr[0] = gt_pn[0]
        pred_PE_arr[0] = gt_pe[0]
        for t in range(1, n):
            step_dt = stride_val * dt_win[t]
            pred_PN_arr[t] = pred_PN_arr[t - 1] + pred_vel[t, 0] * step_dt
            pred_PE_arr[t] = pred_PE_arr[t - 1] + pred_vel[t, 1] * step_dt

        # Per-step position errors
        errs = np.sqrt((pred_PN_arr - gt_pn_np) ** 2 +
                       (pred_PE_arr - gt_pe_np) ** 2)

        results.append(SimulateResult(
            model_name=model_name,
            color=MODEL_PALETTE.get(model_name, "#888888"),
            pred_PN=pred_PN_arr.tolist(),
            pred_PE=pred_PE_arr.tolist(),
            errors=errs.tolist(),
            ade=round(float(errs.mean()), 4),
            fde=round(float(errs[-1]), 4),
            inference_ms=round(inference_ms, 3),
        ))

    return SimulateResponse(
        gt_PN=gt_pn,
        gt_PE=gt_pe,
        total_windows=n,
        results=results,
    )


@app.get("/datasets")
def list_datasets():
    if not os.path.exists(DATA_DIR):
        return {"datasets": []}
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    # Prefer v1.csv as default
    if "v1.csv" in files:
        files.remove("v1.csv")
        files.insert(0, "v1.csv")
    elif "Aligned_Data_preprocessed.csv" in files:
        files.remove("Aligned_Data_preprocessed.csv")
        files.insert(0, "Aligned_Data_preprocessed.csv")
    return {"datasets": files}


@app.get("/dataset/{filename}")
def get_dataset(filename: str):
    """Return raw features + ground-truth positions (backward-compatible)."""
    raw, gt_PN, gt_PE, dt_arr = _load_dataset(filename)
    if not raw:
        raise HTTPException(status_code=422, detail="Empty dataset")
    return {"raw": raw, "dt": dt_arr[0] if dt_arr else 0.1,
            "gtPN": gt_PN, "gtPE": gt_PE}


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset with validation and cleaning."""
    # 1. Check extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, detail="Only .csv files are supported")
    
    # 2. Check MIME type (loose check)
    if file.content_type and file.content_type not in ('text/csv', 'application/vnd.ms-excel', 'application/octet-stream', 'text/plain'):
        raise HTTPException(status_code=422, detail=f"Invalid file type: {file.content_type}. Expected CSV.")
    
    # 3. Read and check size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Maximum is {MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB.")
    
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="File is empty.")
    
    # 4. Save file
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, 'wb') as f:
        f.write(content)
    
    # 5. Validate and clean
    report = _validate_and_clean_csv(save_path)
    report["filename"] = file.filename
    report["status"] = "uploaded"
    
    return report

@app.get("/validate_dataset/{filename}")
def validate_dataset(filename: str):
    """Validate an existing dataset and return a detailed report."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset '{filename}' not found")
    report = _validate_and_clean_csv(path)
    report["filename"] = filename
    return report
