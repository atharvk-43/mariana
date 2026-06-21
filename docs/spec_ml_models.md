# Spec: ML Models
## Files: `backend/ml/features.py`, `isolation_forest.py`, `lstm_ae.py`, `prophet_model.py`, `gat_model.py`, `ensemble.py`

---

## Key Design Principle

All four models are **unsupervised or self-supervised**. None require `is_anomaly` labels for training. This is intentional — it makes the system robust to noisy or uncertain fault injection labels.

---

## File 1: `backend/ml/features.py`

### Purpose
Transform raw telemetry rows into enriched feature vectors for ML models. Called at both training time (batch) and inference time (online, per-tick).

### Input
A pandas DataFrame with **92 raw telemetry columns** (4 identity + 40 per-interface + 18 per-tunnel + 23 NUMERIC_COLS + 3 non-numeric aggregates + 4 labels). For ML, use the 23 `NUMERIC_COLS` aggregate fields. Per-interface fields (8 per iface: `errors_in`, `drops_in`, `utilization_pct`, `queue_depth`, `latency_ms`, `jitter_ms`, `packet_loss_pct`, `link_state`) and per-tunnel fields (3 per tunnel: `loss_pct`, `jitter_ms`, `latency_ms`) are available for granular alerting and feature engineering.

### Feature Groups

```python
NUMERIC_COLS = [
    "bytes_in", "bytes_out", "packets_in", "packets_out",
    "errors_in", "drops_in", "drops_out", "utilization_pct",
    "bgp_sessions_active", "bgp_prefixes_received",
    "bgp_updates_per_min", "bgp_withdrawals_per_min",
    "ospf_spf_runs",
    "ldp_sessions_active", "mpls_lsp_count",
    "mpls_label_table_size", "vpn_routes_count",
    "cpu_load_pct", "memory_used_pct", "queue_depth",
    "latency_ms", "jitter_ms", "packet_loss_pct",
]  # 23 numeric features — these are the base (aggregates from per-interface data)

# Per-interface fields are also available for richer feature engineering:
# eth0_bytes_in, eth0_utilization_pct, eth1_latency_ms, wan0_queue_depth, etc.
# Per-tunnel fields: lsp-b1-dc1_loss_pct, lsp-b1-dc1_latency_ms, etc.

ROLLING_WINDOWS_SEC = [30, 300, 3600]  # 30s, 5min, 1hr
# For each window and each numeric col, compute: mean, std, slope
# slope = (last_value - first_value) / window_duration
# This adds 23 * 3 * 3 = 207 rolling features
# Total feature vector after rolling: 23 + 207 = 230 features
# BUT: only 23-feature vector used for IF and single-row inference
# Rolling features only available when enough history exists (use for LSTM, Prophet)
```

### Functions

```python
def build_feature_matrix(df: pd.DataFrame, per_node: bool = True) -> pd.DataFrame:
    """
    Given a dataframe (all nodes, all times), compute rolling features.
    If per_node=True, group by node_id first before computing rolling stats.
    
    Returns df with original columns + rolling feature columns.
    Rolling feature column naming: f"{col}_mean_{window}s", f"{col}_std_{window}s", f"{col}_slope_{window}s"
    
    For rows where not enough history exists (< window size), fill with current value (mean) or 0 (std/slope).
    
    Note: The input CSV now has 92 columns (per-interface + per-tunnel + NUMERIC_COLS aggregates + labels).
    Use NUMERIC_COLS (23 aggregates) as the base feature set. Per-interface fields (eth0_*, eth1_*, etc.)
    and per-tunnel fields (lsp-*_*) are available for advanced feature engineering.
    """

def get_feature_vector_for_if(row: dict) -> np.ndarray:
    """
    Returns a 23-element float array from a single row dict.
    Used by IsolationForestDetector at inference time.
    Order must match NUMERIC_COLS exactly.
    Handle link_state separately: encode as {"UP": 0, "FLAPPING": 1, "DOWN": 2}
    Include encoded link_state as 24th feature.
    Returns shape (24,).
    """

def get_sequence_for_lstm(
    history_df: pd.DataFrame,
    node_id: str,
    seq_len: int = 60,
    step: int = 1,
) -> np.ndarray:
    """
    Extract a (seq_len, 23) float array from the last seq_len rows for node_id.
    If fewer than seq_len rows exist, zero-pad at the beginning.
    Returns shape (seq_len, 23).
    Used by LSTMAutoencoder at inference time.
    """

def get_graph_features(
    latest_rows: dict[str, dict],  # node_id → latest row dict
) -> np.ndarray:
    """
    Given latest row for each of the 10 nodes, return (10, 20) feature matrix.
    Row order matches NODES list from topology.py.
    Columns are first 20 from NUMERIC_COLS.
    Returns shape (10, 20).
    Used by GATAnomalyDetector at inference time.
    """

def normalize_features(
    X: np.ndarray,
    scaler=None,
    fit: bool = False,
) -> tuple[np.ndarray, object]:
    """
    StandardScaler normalization.
    If fit=True: fit scaler on X, return (X_normalized, fitted_scaler).
    If fit=False: use provided scaler to transform, return (X_normalized, scaler).
    Scaler should be saved alongside model artifacts.
    """
```

---

## File 2: `backend/ml/isolation_forest.py`

### Purpose
Stateless per-datapoint anomaly detector. Fast, no GPU needed. Good baseline.

### Class

```python
class IsolationForestDetector:
    """
    Wrapper around sklearn IsolationForest.
    Trained on normal-period data from telemetry_train.csv.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,  # expected anomaly fraction
        n_estimators: int = 200,
        max_samples: int = 512,
        random_state: int = 42,
    ):
    
    def fit(self, X: np.ndarray) -> None:
        """
        Train on feature matrix X, shape (N, 22).
        X should contain only NORMAL period rows (filter where is_anomaly==False).
        Fits internal sklearn IsolationForest and a StandardScaler.
        Saves scaler internally.
        """
    
    def score(self, x: np.ndarray) -> float:
        """
        Score a single feature vector x, shape (22,).
        Returns anomaly_score ∈ [0, 1] where 1 = most anomalous.
        
        IsolationForest.score_samples() returns ∈ [-0.5, 0] where more negative = more anomalous.
        Normalize to [0,1]: score = 1 - (raw_score + 0.5) / 0.5
        """
    
    def predict(self, x: np.ndarray) -> dict:
        """
        Returns: {"anomaly_score": float, "is_anomaly": bool}
        is_anomaly = True if anomaly_score > 0.6
        """
    
    def save(self, path: str) -> None:
        """Pickle self to path. Save to backend/models/isolation_forest.pkl"""
    
    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Load from pickle file."""
    
    @staticmethod
    def train_from_csv(csv_path: str, model_save_path: str) -> "IsolationForestDetector":
        """
        Convenience method:
        1. Load telemetry_train.csv
        2. Filter rows where is_anomaly==False
        3. Build feature matrix via features.get_feature_vector_for_if()
        4. Fit and save
        """
```

### Training Notes
- Filter `is_anomaly == False` rows for training data
- Expected training data: ~285,000 rows (95% of 300k)
- Training time: < 5 minutes on CPU
- Run locally, not on Kaggle

---

## File 3: `backend/ml/lstm_ae.py`

### Purpose
LSTM Autoencoder trained on **normal sequences only**. Detects pattern-level anomalies (gradual drift, correlated multi-metric shifts) that Isolation Forest misses.

### Architecture

```python
class LSTMEncoder(nn.Module):
    """
    Input: (batch, seq_len, n_features)
    Output: (batch, hidden_dim)  ← latent representation
    
    Architecture:
      LSTM(n_features → hidden_dim=256, num_layers=3, batch_first=True, dropout=0.2)
      Take last hidden state: output[:, -1, :]
    """

class LSTMDecoder(nn.Module):
    """
    Input: (batch, hidden_dim)
    Output: (batch, seq_len, n_features)  ← reconstructed sequence
    
    Architecture:
      Repeat latent vector seq_len times → (batch, seq_len, hidden_dim)
      LSTM(hidden_dim → hidden_dim=256, num_layers=2, batch_first=True)
      Linear(hidden_dim → n_features)
    """

class LSTMAutoencoder(nn.Module):
    """
    Combines Encoder + Decoder.
    
    forward(x):
      x shape: (batch, seq_len=60, n_features=23)
      z = encoder(x)      # (batch, 256)
      x_hat = decoder(z)  # (batch, 60, 23)
      return x_hat
    """
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample mean squared error between x and reconstructed x.
        Shape: (batch,) — one error value per sample.
        """
```

### Training (Kaggle Notebook)

```python
# Hyperparameters
SEQ_LEN = 60          # 60 timesteps = 2 minutes at 2s intervals
N_FEATURES = 23       # NUMERIC_COLS (aggregates from per-interface data)
HIDDEN_DIM = 256
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training data preparation
# 1. Load telemetry_train.csv (92 columns; use only NUMERIC_COLS 23 aggregate fields)
# 2. Filter is_anomaly == False (keep only normal rows)
# 3. Group by node_id, sort by timestamp
# 4. Create sliding windows of seq_len=60 with stride=1
#    → Each window: (60, 23) float tensor, normalized via StandardScaler
# 5. Split 80/20 train/val by time (no shuffle — time ordering preserved)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

# Training:
# for epoch in range(EPOCHS):
#   train_loss = 0
#   for batch in train_loader:
#     optimizer.zero_grad()
#     x_hat = model(batch)
#     loss = criterion(x_hat, batch)
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     optimizer.step()
#     train_loss += loss.item()
#   scheduler.step(val_loss)

# Threshold computation (after training):
# Run validation set through model
# Compute reconstruction_error for each sample
# threshold = np.percentile(val_errors, 95)
# Save threshold alongside model

# Save artifacts:
# torch.save({
#   "model_state": model.state_dict(),
#   "threshold": threshold,
#   "scaler_mean": scaler.mean_,
#   "scaler_std": scaler.scale_,
#   "seq_len": SEQ_LEN,
#   "n_features": N_FEATURES,
# }, "lstm_ae.pt")
```

### Inference Class (for API use)

```python
class LSTMAnomalyDetector:
    """Loads trained model and scores sequences at inference time."""
    
    def __init__(self, model_path: str = "backend/models/lstm_ae.pt"):
        """Load model, threshold, and scaler from saved .pt file."""
    
    def score(self, sequence: np.ndarray) -> dict:
        """
        Input: sequence shape (60, 23) — raw (unnormalized) feature values.
        Steps:
          1. Normalize sequence using loaded scaler
          2. Convert to tensor, add batch dim → (1, 60, 23)
          3. Forward pass → reconstruction
          4. Compute reconstruction error (scalar)
          5. Normalize to [0,1]: lstm_score = min(error / (3 * threshold), 1.0)
        Returns: {"lstm_score": float, "reconstruction_error": float, "is_anomaly": bool}
        is_anomaly = reconstruction_error > threshold
        """
```

---

## File 4: `backend/ml/prophet_model.py`

### Purpose
Per-metric time-series forecasting. Fits one Prophet model per (node_id, metric_name) pair. Detects anomalies when actual value deviates from forecast. Estimates time-to-impact (when forecast predicts a threshold breach).

### Metrics to Forecast (targeted per fault type)

Only lead metrics on nodes that can actually experience each fault. No blanket forecasting.

```python
PROPHET_METRICS = [
    # Congestion precursors — CE nodes (3 nodes × 2 metrics = 6 models)
    ("CE-B1", "utilization_pct",     90.0,  "above"),
    ("CE-B1", "queue_depth",         100.0, "above"),
    ("CE-B2", "utilization_pct",     90.0,  "above"),
    ("CE-B2", "queue_depth",         100.0, "above"),
    ("CE-B3", "utilization_pct",     90.0,  "above"),
    ("CE-B3", "queue_depth",         100.0, "above"),

    # BGP flap precursor — PE nodes (2 nodes × 1 metric = 2 models)
    ("PE-1",  "bgp_updates_per_min", 50.0,  "above"),
    ("PE-2",  "bgp_updates_per_min", 50.0,  "above"),

    # MPLS failure precursors — P nodes (3 nodes × 3 metrics = 9 models)
    ("P-1",   "errors_in",           20.0,  "above"),
    ("P-1",   "cpu_load_pct",        85.0,  "above"),
    ("P-1",   "tunnel_packet_loss_pct", 2.0, "above"),
    ("P-2",   "errors_in",           20.0,  "above"),
    ("P-2",   "cpu_load_pct",        85.0,  "above"),
    ("P-2",   "tunnel_packet_loss_pct", 2.0, "above"),
    ("P-3",   "errors_in",           20.0,  "above"),
    ("P-3",   "cpu_load_pct",        85.0,  "above"),
    ("P-3",   "tunnel_packet_loss_pct", 2.0, "above"),

    # Policy drift precursors — PE nodes (2 nodes × 2 metrics = 4 models)
    ("PE-1",  "cpu_load_pct",        85.0,  "above"),
    ("PE-1",  "vpn_routes_count",    500.0, "above"),
    ("PE-2",  "cpu_load_pct",        85.0,  "above"),
    ("PE-2",  "vpn_routes_count",    500.0, "above"),

    # Systemic latency creep — core & edge nodes (5 nodes × 1 metric = 5 models)
    ("PE-1",  "latency_ms",          150.0, "above"),
    ("PE-2",  "latency_ms",          150.0, "above"),
    ("P-1",   "latency_ms",          50.0,  "above"),
    ("P-2",   "latency_ms",          50.0,  "above"),
    ("P-3",   "latency_ms",          50.0,  "above"),

    # Systemic packet loss — for completeness (5 nodes × 1 metric = 5 models)
    ("PE-1",  "packet_loss_pct",     5.0,   "above"),
    ("PE-2",  "packet_loss_pct",     5.0,   "above"),
    ("P-1",   "packet_loss_pct",     3.0,   "above"),
    ("P-2",   "packet_loss_pct",     3.0,   "above"),
    ("P-3",   "packet_loss_pct",     3.0,   "above"),
]
# Total models: 31 — each (node, metric) pair maps to a specific fault precursor.
# No blanket forecasting. No wasted models on metrics that never breach.
```

### Class

```python
class ProphetForecaster:
    """
    Manages a collection of Prophet models, one per (node_id, metric).
    Models are stored as a dict: models[(node_id, metric)] = fitted_prophet
    """
    
    def __init__(self):
        self.models: dict[tuple, Prophet] = {}
        self.thresholds = {m[0]: (m[1], m[2]) for m in PROPHET_METRICS}
    
    def fit_all(self, df: pd.DataFrame, parallel: bool = True) -> None:
        """
        Fit all 31 Prophet models from historical DataFrame.
        
        For each (node_id, metric) in PROPHET_METRICS:
          1. Filter df for node_id, select [timestamp, metric] → rename to [ds, y]
          2. Create Prophet(
               seasonality_mode='multiplicative',
               daily_seasonality=True,
               weekly_seasonality=False,  # only 7 days of data
               changepoint_prior_scale=0.05,  # conservative
             )
          3. Add custom seasonality: model.add_seasonality(name='diurnal', period=1, fourier_order=5)
          4. model.fit(df_node)
          5. Store in self.models[(node_id, metric)]
        
        If parallel=True: use multiprocessing.Pool(cpu_count()) to fit all 31.
        Expected time: ~2 min (parallel) vs ~12 min (serial) on 8 CPU.
        """
    
    def forecast(
        self,
        node_id: str,
        metric: str,
        horizon_hours: list[float] = [1.0, 3.0, 6.0],
    ) -> dict:
        """
        Returns forecast for the next 1h, 3h, 6h for a given node + metric.
        
        Steps:
          1. Get model for (node_id, metric)
          2. Create future DataFrame (Prophet format) for horizon_hours
          3. model.predict(future)
          4. Return:
          {
            "metric": str,
            "node_id": str,
            "horizon": {
              "1h": {"yhat": float, "yhat_lower": float, "yhat_upper": float},
              "3h": {...},
              "6h": {...},
            },
            "time_to_breach_hours": float | null,  # null if no breach predicted
            "breach_threshold": float,
          }
        """
    
    def estimate_time_to_breach(
        self,
        node_id: str,
        metric: str,
        max_hours: float = 12.0,
    ) -> float | None:
        """
        Forecast at fine granularity (15-min steps) up to max_hours.
        Find earliest timestep where yhat crosses threshold.
        Returns hours until breach, or None if no breach predicted.
        """
    
    def anomaly_score(
        self,
        node_id: str,
        metric: str,
        actual_value: float,
    ) -> float:
        """
        Compare actual_value to the forecast for 'now'.
        Returns prophet_score ∈ [0, 1]:
          - If actual within [yhat_lower, yhat_upper]: score near 0
          - If actual outside bounds: score proportional to deviation
          Formula: score = min(|actual - yhat| / (yhat_upper - yhat_lower + 1e-6), 1.0)
        Returns 0.0 if model not yet fitted for this (node_id, metric).
        """
    
    def get_overall_score(self, node_id: str, current_row: dict) -> dict:
        """
        Aggregate Prophet scores across all metrics for one node.
        Returns:
        {
          "prophet_score": float,   # max anomaly score across all metrics
          "per_metric_scores": dict,  # metric → score
          "earliest_breach_hours": float | None,  # minimum time-to-breach across all metrics
          "breach_metric": str | None,  # which metric will breach first
        }
        """
    
    def save(self, save_dir: str) -> None:
        """
        Serialize all models to save_dir/prophet/.
        Each model saved as: {node_id}__{metric}.pkl using joblib.
        Also save a manifest: manifest.json listing all fitted (node_id, metric) pairs.
        """
    
    @classmethod
    def load(cls, save_dir: str) -> "ProphetForecaster":
        """Load all models from save_dir/prophet/ using manifest."""
```

---

## File 5: `backend/ml/gat_model.py`

### Purpose
Graph Attention Network operating as a **graph autoencoder**. Reconstructs node features from the network topology context. High reconstruction error for a node = that node is in an anomalous state relative to its topological neighbors.

### Architecture

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GATEncoder(nn.Module):
    """
    3-layer Graph Attention Network.
    Input: node features (10 nodes × 20 features) — first 20 NUMERIC_COLS
    Output: node embeddings (10 nodes × latent_dim)
    
    Architecture:
      GATConv(20 → 128, heads=8, dropout=0.2)   → (10, 128*8=1024)
      ELU activation
      GATConv(1024 → 64, heads=4, dropout=0.2)  → (10, 64*4=256)
      ELU activation
      GATConv(256 → latent_dim=32, heads=1)     → (10, 32)
    """

class GATDecoder(nn.Module):
    """
    Linear decoder: reconstructs node features from embeddings.
    Input: (10, latent_dim=32)
    Output: (10, 20)  ← reconstructed node features (first 20 NUMERIC_COLS)
    
    Architecture:
      Linear(32 → 128) + ELU
      Linear(128 → 20)
    """

class GATAutoencoder(nn.Module):
    """
    full GAT graph autoencoder.
    
    forward(data: Data) → reconstructed_features (10, 20)
    data.x: node features (10, 20) — first 20 NUMERIC_COLS
    data.edge_index: adjacency in COO format (2, num_edges)
    """
    
    def reconstruction_error(self, data: Data) -> torch.Tensor:
        """
        Per-node MSE between original and reconstructed features.
        Returns shape (10,) — one error per node.
        """
```

### Training (Kaggle Notebook)

```python
# Hyperparameters
N_NODES = 10
N_FEATURES = 20  # first 20 NUMERIC_COLS (aggregates from per-interface data)
LATENT_DIM = 32
BATCH_SIZE = 64         # number of graph snapshots per batch
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data preparation:
# 1. Load graph_snapshots.pkl (list of snapshot dicts)
# 2. Filter: keep only snapshots where NO node is anomalous (all node_labels == 0)
#    → these are the normal graph states for AE training
# 3. For each snapshot, create torch_geometric Data object:
#    data = Data(
#      x=torch.tensor(snapshot["node_features"], dtype=torch.float32),
#      edge_index=edge_index,  # precomputed COO from topology.EDGES
#    )
# 4. Use DataLoader from torch_geometric

# edge_index computation (precompute once):
# from topology import EDGES, get_node_index
# edges = [(get_node_index(src), get_node_index(dst)) for src, dst, _ in EDGES]
# edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Training loop:
# criterion = nn.MSELoss()
# for epoch in EPOCHS:
#   for batch in loader:
#     optimizer.zero_grad()
#     x_hat = model(batch)
#     loss = criterion(x_hat, batch.x)  # reconstructed vs original
#     loss.backward()
#     optimizer.step()

# Threshold:
# Run validation set (normal graphs only)
# Compute per-node reconstruction error
# threshold = np.percentile(all_node_errors, 95)

# Save:
# torch.save({
#   "model_state": model.state_dict(),
#   "threshold": threshold,
#   "scaler_mean": scaler.mean_,
#   "scaler_std": scaler.scale_,
#   "edge_index": edge_index.numpy(),
# }, "gat.pt")
```

### Inference Class

```python
class GATAnomalyDetector:
    
    def __init__(self, model_path: str = "backend/models/gat.pt"):
        """Load model, threshold, scaler, edge_index from .pt file."""
    
    def score(self, node_features: np.ndarray) -> dict:
        """
        Input: node_features shape (10, 20) — first 20 NUMERIC_COLS for all nodes.
               These are aggregate fields computed from per-interface telemetry.
        Steps:
          1. Normalize using loaded scaler
          2. Create torch_geometric Data object
          3. Forward pass → reconstruction error per node (10,)
          4. Normalize each node's error to [0,1]
        Returns:
        {
          "per_node_scores": dict[node_id → float],  # gat anomaly score per node
          "per_node_anomaly": dict[node_id → bool],  # True if error > threshold
          "network_score": float,  # max across all nodes
        }
        """
```

---

## File 6: `backend/ml/ensemble.py`

### Purpose
Fuse outputs from all 4 models into a single risk score, health state, and time-to-impact estimate for each node.

### Ensemble Weights

```python
ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.20,
    "lstm_ae":          0.30,
    "prophet":          0.30,
    "gat":              0.20,
}
# These weights can be tuned. LSTM and Prophet weighted higher as they capture temporal patterns.
```

### Class

```python
class EnsembleDetector:
    """
    Aggregates all 4 model outputs per node.
    Models should all be loaded and ready before instantiating this class.
    """
    
    def __init__(
        self,
        if_detector: IsolationForestDetector,
        lstm_detector: LSTMAnomalyDetector,
        prophet_forecaster: ProphetForecaster,
        gat_detector: GATAnomalyDetector,
    ):
    
    def predict(
        self,
        node_id: str,
        current_row: dict,
        history_df: pd.DataFrame,
        all_node_latest: dict[str, dict],
    ) -> dict:
        """
        Run full inference pipeline for one node.
        
        Steps:
        1. IF: score = if_detector.score(get_feature_vector_for_if(current_row))
        2. LSTM: score = lstm_detector.score(get_sequence_for_lstm(history_df, node_id))
        3. Prophet: result = prophet_forecaster.get_overall_score(node_id, current_row)
        4. GAT: gat_scores = gat_detector.score(get_graph_features(all_node_latest))
               node_gat_score = gat_scores["per_node_scores"][node_id]
        5. Ensemble: final_risk = weighted sum of all 4 scores
        6. Health state classification (see below)
        7. Time-to-impact from Prophet
        
        Returns:
        {
          "node_id": str,
          "timestamp": str,
          "risk_score": float,          # 0–100 (final_risk * 100)
          "health_state": str,          # "NORMAL" | "WARNING" | "CRITICAL"
          "time_to_impact_hours": float | None,
          "breach_metric": str | None,
          "model_scores": {
            "isolation_forest": float,
            "lstm_ae": float,
            "prophet": float,
            "gat": float,
          },
          "metric_health": dict[str, str],  # per-metric NORMAL/WARNING/CRITICAL
        }
        """
    
    def _classify_health(self, risk_score: float) -> str:
        """
        risk_score ∈ [0, 1]
        < 0.35: NORMAL
        0.35–0.65: WARNING
        > 0.65: CRITICAL
        """
    
    def _classify_metric_health(self, current_row: dict) -> dict[str, str]:
        """
        Rule-based per-metric health classification.
        
        Thresholds:
        latency_ms:           WARNING > 50, CRITICAL > 150
        packet_loss_pct:      WARNING > 1.0, CRITICAL > 5.0
        utilization_pct:      WARNING > 75, CRITICAL > 90
        cpu_load_pct:         WARNING > 70, CRITICAL > 85
        bgp_updates_per_min:  WARNING > 15, CRITICAL > 50
        bgp_sessions_active:  WARNING < total-1, CRITICAL if 0
        errors_in:            WARNING > 5, CRITICAL > 20
        memory_used_pct:      WARNING > 80, CRITICAL > 90
        queue_depth:          WARNING > 40, CRITICAL > 100
        link_state:           FLAPPING → WARNING, DOWN → CRITICAL
        """
    
    def predict_all_nodes(
        self,
        current_rows: list[dict],
        history_df: pd.DataFrame,
    ) -> list[dict]:
        """
        Run predict() for all 10 nodes and return list of results.
        Passes all_node_latest to GAT (needs all nodes simultaneously).
        """
```

---

## Model File Locations

```
backend/models/
├── isolation_forest.pkl       # ~50MB
├── lstm_ae.pt                 # ~20MB — download from Kaggle
├── gat.pt                     # ~5MB — download from Kaggle
├── prophet/
│   ├── manifest.json
│   ├── PE-1__bgp_updates_per_min.pkl   # BGP flap precursor
│   ├── CE-B1__utilization_pct.pkl      # Congestion precursor
│   ├── P-1__errors_in.pkl              # MPLS failure precursor
│   ├── ... (31 files total — targeted lead metrics only)
└── scaler_if.pkl              # StandardScaler for IF
```

---

## Dependencies (add to requirements.txt)

```
torch>=2.0.0
torch-geometric>=2.3.0
prophet>=1.1.4
scikit-learn>=1.9.0
joblib>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

For Kaggle notebooks, also install:
```
!pip install torch-geometric prophet
```
