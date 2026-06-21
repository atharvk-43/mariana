import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.network_gen import NUMERIC_COLS
from ..data.topology import NODES, NODE_INDEX

ROLLING_WINDOWS_SEC = [30, 300, 3600]

LINK_STATE_MAP = {"UP": 0, "FLAPPING": 1, "DOWN": 2, "N/A": 0}


def build_feature_matrix(df: pd.DataFrame, per_node: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    group_key = "node_id" if per_node else None
    groups = result.groupby(group_key) if group_key else [(None, result)]

    for _, group_df in groups:
        mask = result.index.isin(group_df.index) if group_key else slice(None)
        for col in NUMERIC_COLS:
            if col not in group_df.columns:
                continue
            for w in ROLLING_WINDOWS_SEC:
                steps = max(1, w // 2)
                rolled = group_df[col].rolling(window=steps, min_periods=1)
                result.loc[mask, f"{col}_mean_{w}s"] = rolled.mean()
                result.loc[mask, f"{col}_std_{w}s"] = rolled.std().fillna(0)
                result.loc[mask, f"{col}_slope_{w}s"] = (
                    (group_df[col] - group_df[col].shift(steps)).fillna(0) / w
                )

    return result


def get_feature_vector_for_if(row: dict) -> np.ndarray:
    vals = [float(row.get(c, 0.0)) for c in NUMERIC_COLS]
    ls = LINK_STATE_MAP.get(row.get("link_state", "UP"), 0)
    return np.array(vals + [float(ls)], dtype=np.float32)


def get_sequence_for_lstm(
    history_df: pd.DataFrame,
    node_id: str,
    seq_len: int = 60,
    step: int = 1,
) -> np.ndarray:
    node_df = history_df[history_df["node_id"] == node_id].sort_values("timestamp")
    values = node_df[NUMERIC_COLS].values
    if len(values) >= seq_len:
        seq = values[-seq_len::step][:seq_len]
    else:
        pad = np.zeros((seq_len - len(values), len(NUMERIC_COLS)), dtype=np.float32)
        seq = np.vstack([pad, values[::step]]) if len(values) > 0 else pad
    return np.array(seq, dtype=np.float32)


def get_graph_features(latest_rows: dict[str, dict]) -> np.ndarray:
    features = np.zeros((len(NODES), 20), dtype=np.float32)
    for i, nid in enumerate(NODE_INDEX):
        row = latest_rows.get(nid, {})
        for j, col in enumerate(NUMERIC_COLS[:20]):
            features[i, j] = float(row.get(col, 0.0))
    return features


def normalize_features(
    X: np.ndarray,
    scaler=None,
    fit: bool = False,
) -> tuple[np.ndarray, object]:
    if fit or scaler is None:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
    else:
        X_norm = scaler.transform(X)
    return X_norm.astype(np.float32), scaler
