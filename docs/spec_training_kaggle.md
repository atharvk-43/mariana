# Spec: Kaggle Training Notebooks
## Files: `nbs/train_lstm_ae_kaggle.ipynb`, `nbs/train_gat_kaggle.ipynb`, `nbs/train_if_kaggle.ipynb`, `nbs/train_prophet_kaggle.ipynb`

## Status: ✅ Implemented — spec is mostly superseded by actual notebooks

> **Updates vs this spec:**
> - 4 notebooks exist (not 2): LSTM AE, GAT, IF, Prophet
> - File paths are `nbs/`, not `backend/training/`
> - Actual notebooks include Optuna HPO, DataParallel, per-node evaluation, model+scaler download — none in this spec
> - LSTM AE: actual uses N_FEATURES=21, HIDDEN_DIM=128, 3 enc/2 dec layers, DataParallel, Optuna trial for lr/hidden_dim/dropout, 100 epochs, checkpoints at best val_loss, Kaggle T4×2 GPU
> - GAT AE: actual uses N_FEATURES=20, LATENT_DIM=16, 2 GAT layers (not 3), Optuna trial for lr/latent_dim/dropout, 50 epochs, 95th-pct threshold, Kaggle T4 GPU
> - IF: not in original spec — added via Optuna HPO over 15 trials, 70 features (not 24), Kaggle CPU
> - Prophet: not in original spec — grid search over changepoint_prior_scale (5 values), seasonality_prior_scale=10.0, 30 models, Kaggle CPU
> - Kaggle dataset name: "MPLS Network Telemetry — PS-13 NOC Anomaly Detection"
> - PyTorch 2.6 fix: verify blocks need `weights_only=False`

---

## Overview

These notebooks are designed to run on Kaggle T4 GPU. They are self-contained — all required code is inline (don't import from the project — Kaggle can't see your local files).

**Workflow:**
1. Run `generate_dataset.py` locally → produces `telemetry_train.csv` and `graph_snapshots.pkl`
2. Upload both files to a new Kaggle Dataset (private)
3. Create a new Kaggle Notebook, add the dataset
4. Copy/paste notebook cells (see below)
5. Enable GPU (T4 x2 recommended)
6. Run all cells → download `lstm_ae.pt` and `gat.pt`
7. Place both files in `backend/models/`

---

## Notebook 1: `train_lstm_kaggle.ipynb`

### Cell 1 — Install dependencies
```python
!pip install -q prophet  # only if needed, torch already on Kaggle
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Cell 2 — Constants
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Config ----
SEQ_LEN     = 60        # 60 timesteps = 2 minutes
N_FEATURES  = 21        # numeric telemetry features
HIDDEN_DIM  = 256
ENC_LAYERS  = 3
DEC_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 256
EPOCHS      = 80
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Which columns to use as features (order matters — must match inference)
NUMERIC_COLS = [
    "bytes_in", "bytes_out", "packets_in", "packets_out",
    "errors_in", "drops_in", "drops_out", "utilization_pct",
    "bgp_sessions_active", "bgp_prefixes_received",
    "bgp_updates_per_min", "bgp_withdrawals_per_min", "ospf_spf_runs",
    "cpu_load_pct", "memory_used_pct", "queue_depth",
    "latency_ms", "jitter_ms", "packet_loss_pct",
    "tunnel_packet_loss_pct", "ipsec_rekeyed_last_hr",
]

print(f"Features: {len(NUMERIC_COLS)}, Device: {DEVICE}")
```

### Cell 3 — Load and prepare data
```python
# Load CSV (update path to match your Kaggle dataset path)
df = pd.read_csv("/kaggle/input/your-dataset-name/telemetry_train.csv")
print(f"Total rows: {len(df)}")

# Keep only normal rows for AE training
df_normal = df[df["is_anomaly"] == False].copy()
print(f"Normal rows (for training): {len(df_normal)}")
print(f"Anomaly rate: {(df['is_anomaly'].sum() / len(df) * 100):.1f}%")

# Sort by node then time (important for sequence integrity)
df_normal = df_normal.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

# Fit scaler on normal data
scaler = StandardScaler()
df_normal[NUMERIC_COLS] = scaler.fit_transform(df_normal[NUMERIC_COLS])

print(f"Scaler fitted. Mean sample: {scaler.mean_[:3]}")
```

### Cell 4 — Create sequence dataset
```python
class SequenceDataset(Dataset):
    def __init__(self, df, seq_len, numeric_cols):
        self.sequences = []
        
        for node_id, group in df.groupby("node_id"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            features = group[numeric_cols].values.astype(np.float32)
            
            # Create sliding windows with stride 1
            for i in range(len(features) - seq_len + 1):
                seq = features[i:i+seq_len]
                self.sequences.append(seq)
        
        self.sequences = np.array(self.sequences)
        print(f"Total sequences: {len(self.sequences)}, shape: {self.sequences.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)

dataset = SequenceDataset(df_normal, SEQ_LEN, NUMERIC_COLS)

# 80/20 split by index (respects time ordering)
split = int(0.8 * len(dataset))
train_ds = torch.utils.data.Subset(dataset, range(split))
val_ds   = torch.utils.data.Subset(dataset, range(split, len(dataset)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
```

### Cell 5 — Model definition
```python
class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # last layer hidden state: (batch, hidden_dim)

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, n_features, seq_len, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)
    
    def forward(self, z):
        # z: (batch, hidden_dim)
        # Repeat z across seq_len
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        out, _ = self.lstm(z_repeated)
        return self.output_layer(out)  # (batch, seq_len, n_features)

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim, enc_layers, dec_layers, seq_len, dropout):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, enc_layers, dropout)
        self.decoder = LSTMDecoder(hidden_dim, n_features, seq_len, dec_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def reconstruction_error(self, x):
        with torch.no_grad():
            x_hat = self.forward(x)
            error = ((x - x_hat) ** 2).mean(dim=[1, 2])  # (batch,)
        return error

model = LSTMAutoencoder(N_FEATURES, HIDDEN_DIM, ENC_LAYERS, DEC_LAYERS, SEQ_LEN, DROPOUT).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Cell 6 — Training loop
```python
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
criterion = nn.MSELoss()

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        x_hat = model(batch)
        loss = criterion(x_hat, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            x_hat = model(batch)
            val_loss += criterion(x_hat, batch).item()
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
```

### Cell 7 — Compute threshold and save
```python
# Compute reconstruction error on validation set
model.eval()
all_errors = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(DEVICE)
        errors = model.reconstruction_error(batch)
        all_errors.extend(errors.cpu().numpy())

all_errors = np.array(all_errors)
threshold = float(np.percentile(all_errors, 95))
print(f"Reconstruction error threshold (95th pct): {threshold:.6f}")
print(f"Error distribution: min={all_errors.min():.4f}, mean={all_errors.mean():.4f}, max={all_errors.max():.4f}")

# Save everything needed for inference
torch.save({
    "model_state": model.state_dict(),
    "threshold": threshold,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std": scaler.scale_.tolist(),
    "seq_len": SEQ_LEN,
    "n_features": N_FEATURES,
    "hidden_dim": HIDDEN_DIM,
    "enc_layers": ENC_LAYERS,
    "dec_layers": DEC_LAYERS,
    "dropout": DROPOUT,
    "numeric_cols": NUMERIC_COLS,
}, "/kaggle/working/lstm_ae.pt")

print("Saved: lstm_ae.pt")
print(f"File size: {os.path.getsize('/kaggle/working/lstm_ae.pt') / 1e6:.1f} MB")
```

---

## Notebook 2: `train_gat_kaggle.ipynb`

### Cell 1 — Install
```python
!pip install -q torch-geometric
import torch
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv
print(f"PyG version: {torch_geometric.__version__}, GPU: {torch.cuda.is_available()}")
```

### Cell 2 — Constants
```python
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

N_NODES     = 10
N_FEATURES  = 20    # numeric features for GAT (excludes ipsec_rekeyed_last_hr)
LATENT_DIM  = 32
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Node order (MUST match topology.py NODES list order)
NODE_ORDER = [
    "pe-router-1", "pe-router-2", "p-router-1", "p-router-2", "p-router-3",
    "ce-branch-1", "ce-branch-2", "ce-branch-3", "ce-dc-1", "ce-dc-2",
]

# Edges as (src_idx, dst_idx) — must match topology.EDGES order
EDGES = [
    (5, 0), (6, 0), (7, 1),   # CE branches → PE
    (0, 2), (1, 2),             # PE → P-1
    (2, 3), (2, 4),             # P-1 → P-2, P-3
    (3, 8), (4, 9),             # P → DC CE
    (0, 1),                     # PE inter-link
]
edge_index = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
# Make undirected
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
print(f"edge_index shape: {edge_index.shape}")
```

### Cell 3 — Load snapshots
```python
with open("/kaggle/input/your-dataset-name/graph_snapshots.pkl", "rb") as f:
    snapshots = pickle.load(f)

print(f"Total snapshots: {len(snapshots)}")

# Filter: only snapshots where ALL nodes are normal
normal_snapshots = [s for s in snapshots if all(s["node_labels"] == 0)]
print(f"Normal snapshots (for AE training): {len(normal_snapshots)}")
print(f"Anomaly snapshot rate: {(1 - len(normal_snapshots)/len(snapshots))*100:.1f}%")

# Fit scaler on normal node features
all_features = np.vstack([s["node_features"] for s in normal_snapshots])
scaler = StandardScaler()
scaler.fit(all_features)

# Build PyG Data objects
def make_data(snapshot):
    x = scaler.transform(snapshot["node_features"])
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
    )

normal_data = [make_data(s) for s in normal_snapshots]

# 80/20 split
split = int(0.8 * len(normal_data))
train_data = normal_data[:split]
val_data   = normal_data[split:]

train_loader = GeoDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = GeoDataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
```

### Cell 4 — Model definition
```python
import torch.nn as nn
import torch_geometric

class GATEncoder(nn.Module):
    def __init__(self, n_features, latent_dim):
        super().__init__()
        self.conv1 = GATConv(n_features, 128, heads=8, dropout=0.2, concat=True)
        self.conv2 = GATConv(128*8, 64,  heads=4, dropout=0.2, concat=True)
        self.conv3 = GATConv(64*4,  latent_dim, heads=1, concat=False)
        self.act   = nn.ELU()
        self.drop  = nn.Dropout(0.2)
    
    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.drop(x)
        x = self.conv3(x, edge_index)
        return x   # (N_nodes, latent_dim)

class GATDecoder(nn.Module):
    def __init__(self, latent_dim, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, n_features),
        )
    
    def forward(self, z):
        return self.net(z)   # (N_nodes, n_features)

class GATAutoencoder(nn.Module):
    def __init__(self, n_features, latent_dim):
        super().__init__()
        self.encoder = GATEncoder(n_features, latent_dim)
        self.decoder = GATDecoder(latent_dim, n_features)
    
    def forward(self, data):
        z = self.encoder(data.x, data.edge_index)
        return self.decoder(z)

model = GATAutoencoder(N_FEATURES, LATENT_DIM).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Cell 5 — Training
```python
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        x_hat = model(batch)
        loss = criterion(x_hat, batch.x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            x_hat = model(batch)
            val_loss += criterion(x_hat, batch.x).item()
    
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss/len(train_loader):.6f} | Val: {val_loss/len(val_loader):.6f}")
```

### Cell 6 — Threshold and save
```python
import os

# Compute per-node reconstruction error on val set
model.eval()
all_node_errors = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(DEVICE)
        x_hat = model(batch)
        errors = ((batch.x - x_hat) ** 2).mean(dim=1)  # (N_nodes * batch_size,)
        all_node_errors.extend(errors.cpu().numpy())

all_node_errors = np.array(all_node_errors)
threshold = float(np.percentile(all_node_errors, 95))
print(f"Per-node threshold: {threshold:.6f}")

torch.save({
    "model_state": model.state_dict(),
    "threshold": threshold,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std": scaler.scale_.tolist(),
    "edge_index": edge_index.numpy().tolist(),
    "n_features": N_FEATURES,
    "latent_dim": LATENT_DIM,
    "node_order": NODE_ORDER,
}, "/kaggle/working/gat.pt")

print(f"Saved: gat.pt ({os.path.getsize('/kaggle/working/gat.pt')/1e6:.1f} MB)")
```

---

## After Kaggle Training

1. Go to Kaggle → Output tab → Download `lstm_ae.pt` and `gat.pt`
2. Place both in `backend/models/`
3. Verify: `python -c "import torch; d = torch.load('backend/models/lstm_ae.pt'); print(d.keys())"`
4. Expected keys: `model_state`, `threshold`, `scaler_mean`, `scaler_std`, `seq_len`, `n_features`, ...
