import numpy as np
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    torch = None
    nn = None
    TORCH_OK = False

try:
    if TORCH_OK:
        from torch_geometric.nn import GATConv
    else:
        GATConv = None
except ImportError:
    GATConv = None


if TORCH_OK and GATConv is not None:
    class GATEncoder(nn.Module):
        def __init__(self, n_features: int = 20, latent_dim: int = 32):
            if GATConv is None:
                raise ImportError("torch_geometric is required for GAT model")
            super().__init__()
            self.conv1 = GATConv(n_features, 128, heads=8, dropout=0.2)
            self.conv2 = GATConv(128 * 8, 64, heads=4, dropout=0.2)
            self.conv3 = GATConv(64 * 4, latent_dim, heads=1)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return self.conv3(x, edge_index)


    class GATDecoder(nn.Module):
        def __init__(self, n_features: int = 20, latent_dim: int = 32):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 128)
            self.fc2 = nn.Linear(128, n_features)

        def forward(self, z):
            return self.fc2(self.fc1(z).relu())


    class GATAutoencoder(nn.Module):
        def __init__(self, n_features: int = 20, latent_dim: int = 32):
            super().__init__()
            self.encoder = GATEncoder(n_features, latent_dim)
            self.decoder = GATDecoder(n_features, latent_dim)

        def forward(self, x, edge_index):
            z = self.encoder(x, edge_index)
            return self.decoder(z)

        def reconstruction_error(self, x, edge_index):
            x_hat = self.forward(x, edge_index)
            return torch.mean((x - x_hat) ** 2, dim=1)


    class GATAnomalyDetector:
        def __init__(self, model_path: Optional[str] = None):
            self.available = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model: Optional[GATAutoencoder] = None
            self.scaler = None
            self.edge_index = None
            self.threshold = 0.0
            if model_path:
                self.load(model_path)

        def score(self, node_features: np.ndarray) -> dict[str, Any]:
            if self.model is None or self.edge_index is None:
                return {"per_node_scores": {}, "per_node_anomaly": {}, "network_score": 0.0}
            self.model.eval()
            x = torch.from_numpy(node_features).float().to(self.device)
            ei = torch.tensor(self.edge_index, dtype=torch.long).to(self.device)
            with torch.no_grad():
                errors = self.model.reconstruction_error(x, ei).cpu().numpy()
            max_err = float(errors.max())
            threshold = self.threshold if self.threshold > 0 else float(np.percentile(errors, 90))
            from ..data.topology import NODE_IDS
            per_node_scores: Dict[str, float] = {}
            per_node_anomaly: Dict[str, bool] = {}
            for i, nid in enumerate(NODE_IDS):
                raw = float(errors[i])
                per_node_scores[nid] = min(raw / (threshold * 3 + 1e-8), 1.0)
                per_node_anomaly[nid] = raw > threshold
            return {
                "per_node_scores": per_node_scores,
                "per_node_anomaly": per_node_anomaly,
                "network_score": max_err,
            }

        def save(self, path: str) -> None:
            if self.model is None:
                raise RuntimeError("No GAT model available to save")
            torch.save({
                "model_state": self.model.state_dict(),
                "scaler": self.scaler,
                "edge_index": self.edge_index,
                "threshold": self.threshold,
            }, path)

        def load(self, path: str) -> None:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = GATAutoencoder().to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.scaler = checkpoint.get("scaler")
            self.edge_index = checkpoint.get("edge_index")
            self.threshold = checkpoint.get("threshold", 0.0)

else:
    class GATEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x, edge_index):
            return x


    class GATDecoder:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, z):
            return z


    class GATAutoencoder:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x, edge_index):
            return x

        def reconstruction_error(self, x, edge_index):
            return np.zeros(x.shape[0], dtype=np.float32)


    class GATAnomalyDetector:
        def __init__(self, model_path: Optional[str] = None):
            self.available = False
            self.model = None
            self.scaler = None
            self.edge_index = None
            self.threshold = 0.0

        def score(self, node_features: np.ndarray) -> dict[str, Any]:
            return {"per_node_scores": {}, "per_node_anomaly": {}, "network_score": 0.0}

        def save(self, path: str) -> None:
            raise RuntimeError("PyTorch or torch_geometric is not installed")

        def load(self, path: str) -> None:
            raise RuntimeError("PyTorch or torch_geometric is not installed")
