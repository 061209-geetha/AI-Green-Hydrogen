from __future__ import annotations
import json, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def config(path=None):
    p = Path(path) if path else Path(__file__).resolve().parents[1] / 'configs/default.yaml'
    with p.open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, (Path,)): return str(o)
        if isinstance(o, (pd.Timestamp,)): return o.isoformat()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(f'Cannot serialize {type(o)}')
    path.write_text(json.dumps(obj, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def checksum(path, chunk=1048576):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()


def save_table(df, base):
    p = Path(base)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p.with_suffix('.csv'), index=False)
    try:
        df.to_parquet(p.with_suffix('.parquet'), index=False)
    except (ImportError, ValueError):
        pass


def safe_mape(y, yhat, eps=1e-8):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    eligible = np.abs(y) > eps
    return float(100 * np.mean(np.abs((y[eligible] - yhat[eligible]) / y[eligible]))) if eligible.any() else float('nan')
