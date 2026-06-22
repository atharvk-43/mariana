# PS-13 Model Training Results

> Last updated: 2026-06-21

---

## Summary

| Model | AUROC | AP | F1 | Training Platform | Status |
|-------|-------|----|----|-------------------|--------|
| **Isolation Forest** | 0.898 | TBD | TBD | Kaggle CPU | ⏳ HPO complete, final training pending |
| **LSTM Autoencoder** | 0.664 | 0.621 | 0.738 | Kaggle GPU (T4×2) | ⏳ Re-submitted with 200 epochs |
| **Prophet** | N/A (MAPE) | — | — | Kaggle CPU | ✅ 30 models trained |
| **GAT Autoencoder** | **0.941** | **0.875** | **0.893** | Kaggle GPU (P100) | ✅ Downloaded |

---

## 1. Isolation Forest

**HPO**: 15 trials, 5 params (contamination, n_estimators, max_samples, max_features, bootstrap)

**Best trial 5**:
```
contamination=0.0078, n_estimators=200, max_samples=0.5,
max_features=0.99, bootstrap=True
AUROC=0.898, AP=0.665
```

**Key insight**: `bootstrap=True` with high `max_features=0.99` produces best separation — IF needs stochastic diversity and full feature visibility.

**Fault coverage**: Best at density-based outliers. Complements AE models.

---

## 2. LSTM Autoencoder

**HPO**: 15 trials, 6 params (hidden_dim, num_layers_enc, num_layers_dec, lr, dropout, weight_decay)

**Best params**: `hidden_dim=256, num_layers_enc=3, num_layers_dec=1, lr=0.00066, dropout=0.117, wd=5.4e-5`

### Per-Fault-Type Performance (max-over-feature scoring)

| Fault Type | Mean Err (×normal) | Median (×normal) | Detection Rate |
|------------|-------------------|------------------|---------------|
| BGP Flap | 246,867× | 25,542× | 72.7% |
| MPLS Failure | 8,571,000× | 3,856× | 64.2% |
| Congestion | 2,118× | **2.2×** | 49.6% |

**Key finding**: LSTM AE is strong on spike faults (BGP, MPLS) but **congestion is invisible** — expected for autoencoders. Ensemble will compensate via IF + Prophet.

**Limitation**: Sequence label dilution (60-timestep windows labeled anomalous if any point is anomalous → average MSE drowns single bad frames). Max-over-feature scoring mitigates but doesn't fully solve this.

---

## 3. Prophet

**HPO**: Grid search, 5 changepoint_prior_scale values, seasonality_prior_scale fixed at 10.0

**Best**: `changepoint_prior_scale=0.01, MAPE=7.25%`

### Per-Metric MAPE

| Metric | MAPE Range | Quality |
|--------|-----------|---------|
| Congestion (utilization_pct) | 1.89% – 7.44% | ✅ Excellent |
| MPLS Failure (latency_ms) | 5.46% – 17.31% | ✅ Good |
| BGP Flap (bgp_updates_per_min) | 42% – 45% | ⚠️ Cosmetic (bursty metric) |

**Best fits**: P-3 (congestion MAPE=1.89%), P-2 (congestion MAPE=2.53%)
**Worst fit**: P-3 (MPLS failure MAPE=17.31%)

**Note**: BGP MAPE is inflated due to zero-inflated metric (most values are 0). Absolute residual still detects spikes. Log-transform option available via `use_log_transform` flag in manifest for cleaner reporting.

**30 models** trained (10 nodes × 3 lead metrics). Saved to `prophet/` with `manifest.json`.

---

## 4. GAT Autoencoder

**HPO**: 25 trials, 8 params (latent_dim, hidden_1, hidden_2, heads_1, heads_2, lr, dropout, weight_decay)

**Best trial 21**: `latent_dim=64, hidden_1=64, hidden_2=32, heads_1=8, heads_2=4, lr=0.00235, dropout=0.137, wd=2.78e-5`

### Per-Node Performance

| Node | AUROC | AP | F1 | Anomalies |
|------|-------|----|----|-----------|
| PE-1 | 0.941 | 0.816 | 0.819 | 62 |
| PE-2 | 0.930 | 0.903 | 0.933 | 40 |
| P-1 | 0.938 | 0.903 | 0.933 | 40 |
| P-2 | 0.970 | 0.920 | 0.919 | 40 |
| P-3 | 0.956 | 0.910 | 0.933 | 40 |
| CE-B1 | 0.919 | 0.905 | 0.933 | 40 |
| CE-B2 | 0.866 | 0.762 | 0.819 | 62 |
| CE-B3 | 0.930 | 0.907 | 0.919 | 40 |
| CE-DC1 | 0.988 | 0.944 | 0.892 | 30 |
| CE-DC2 | **0.997** | **0.954** | 0.892 | 30 |

**Network-Level**: AUROC=0.902, AP=0.777, Snapshot accuracy=97.36%

**Best at**: DC nodes (CE-DC2 AUROC=0.997). Struggles with CE-B2 (0.866, likely due to higher anomaly concentration — 62 vs 30-40).

---

## 5. Ensemble Strategy

| Model | Weight Rationale | Fault Coverage |
|-------|-----------------|----------------|
| **GAT** (0.94 AUROC) | Highest score, topology-aware | Spiky topological faults |
| **IF** (0.90 AUROC) | Density-based, fast | Congestion, multi-dim shifts |
| **LSTM AE** (0.66→0.70) | Temporal patterns | BGP flap, MPLS failure |
| **Prophet** | Forecast residuals | Sustained trends, level shifts |

Proposed weights (to be validated):
```
risk_score = 0.35 × GAT + 0.30 × IF + 0.20 × LSTM + 0.15 × Prophet
```

---

## 6. Model Artifacts

| Model | File(s) | Size | Location |
|-------|---------|------|----------|
| Isolation Forest | `isolation_forest.pkl`, `if_scaler.pkl` | ~50 MB | `src/models/` |
| LSTM Autoencoder | `lstm_ae.pt`, `lstm_scaler.pkl` | ~15 MB | `src/models/` |
| Prophet | 30 × `.pkl`, `manifest.json` | ~150 MB | `src/models/prophet/` |
| GAT Autoencoder | `gat.pt`, `gat_scaler.pkl` | ~2 MB | `src/models/` |
