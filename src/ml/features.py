import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.network_gen import NUMERIC_COLS
from data.topology import NODES, NODE_INDEX

ROLLING_WINDOWS_SEC = [30, 300, 3600]

LINK_STATE_MAP = {"UP": 0, "FLAPPING": 1, "DOWN": 2, "N/A": 0}


def build_feature_matrix(df: pd.DataFrame, per_node: bool = True) -> pd.DataFrame:
    """Compute rolling mean/std/slope features for each numeric column.

    Builds all new columns in a single pd.concat to avoid DataFrame
    fragmentation (which causes PerformanceWarning and ~40% slowdown when
    using iterative result.loc[] assignments).
    """
    if df.empty:
        return df

    group_key = "node_id" if per_node else None
    groups = df.groupby(group_key) if group_key else [(None, df)]

    # Collect new columns per group, then concat once at the end
    new_col_frames = []

    for _, group_df in groups:
        new_cols: dict[str, pd.Series] = {}
        for col in NUMERIC_COLS:
            if col not in group_df.columns:
                continue
            for w in ROLLING_WINDOWS_SEC:
                steps = max(1, w // 2)
                rolled = group_df[col].rolling(window=steps, min_periods=1)
                new_cols[f"{col}_mean_{w}s"] = rolled.mean()
                new_cols[f"{col}_std_{w}s"] = rolled.std().fillna(0)
                new_cols[f"{col}_slope_{w}s"] = (
                    (group_df[col] - group_df[col].shift(steps)).fillna(0) / w
                )
        new_col_frames.append(pd.DataFrame(new_cols, index=group_df.index))

    # Single concat across all groups, then join onto original df
    all_new = pd.concat(new_col_frames).sort_index()
    return pd.concat([df, all_new], axis=1)


def get_feature_matrix_for_if(df: pd.DataFrame) -> np.ndarray:
    """Build IF feature matrix directly from a DataFrame (vectorised, no dict loop).

    Uses NUMERIC_COLS + link_state encoding + any 30s rolling features that
    are already present in the DataFrame.  Much faster and more memory-efficient
    than calling get_feature_vector_for_if() row-by-row.
    """
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    X = df[cols].values.astype(np.float32)

    # Encode link_state
    ls = df["link_state"].map(LINK_STATE_MAP).fillna(0).values.reshape(-1, 1).astype(np.float32)
    X = np.hstack([X, ls])

    # Append 30s rolling features if they exist
    temporal_cols = []
    for col in NUMERIC_COLS:
        for suffix in (f"{col}_mean_30s", f"{col}_slope_30s"):
            if suffix in df.columns:
                temporal_cols.append(suffix)
    if temporal_cols:
        T = df[temporal_cols].values.astype(np.float32)
        X = np.hstack([X, T])

    return X


def get_feature_vector_for_if(row: dict) -> np.ndarray:
    """Build IF feature vector from a single row dict (used for inference).

    Uses raw NUMERIC_COLS plus link_state encoding.
    If rolling features are present (i.e. build_feature_matrix was called
    upstream), appends the 30-second mean and slope for each metric — this
    dramatically improves anomaly separability since faults produce distinct
    slope and variance signatures over time.
    """
    vals = [float(row.get(c, 0.0)) for c in NUMERIC_COLS]
    ls = LINK_STATE_MAP.get(row.get("link_state", "UP"), 0)
    base = vals + [float(ls)]

    # Append temporal features if they exist in the row (optional enrichment)
    temporal = []
    for col in NUMERIC_COLS:
        mean_key = f"{col}_mean_30s"
        slope_key = f"{col}_slope_30s"
        if mean_key in row:
            temporal.append(float(row[mean_key]))
            temporal.append(float(row.get(slope_key, 0.0)))

    return np.array(base + temporal, dtype=np.float32)



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
