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


if TORCH_OK:
    class LSTMEncoder(nn.Module):
        def __init__(self, n_features: int = 23, hidden_dim: int = 256, num_layers: int = 3):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, hidden_dim, num_layers,
                batch_first=True, dropout=0.2 if num_layers > 1 else 0,
            )

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return h_n[-1]


    class LSTMDecoder(nn.Module):
        def __init__(self, n_features: int = 23, hidden_dim: int = 256, num_layers: int = 2, seq_len: int = 60):
            super().__init__()
            self.seq_len = seq_len
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True, dropout=0.2 if num_layers > 1 else 0,
            )
            self.linear = nn.Linear(hidden_dim, n_features)

        def forward(self, z):
            z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.lstm(z)
            return self.linear(out)


    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int = 23, hidden_dim: int = 256, seq_len: int = 60):
            super().__init__()
            self.encoder = LSTMEncoder(n_features, hidden_dim)
            self.decoder = LSTMDecoder(n_features, hidden_dim, seq_len=seq_len)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

        def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
            x_hat = self.forward(x)
            return torch.mean((x - x_hat) ** 2, dim=(1, 2))


    class LSTMAnomalyDetector:
        def __init__(self, model_path: Optional[str] = None):
            self.available = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model: Optional[LSTMAutoencoder] = None
            self.scaler = None
            self.threshold = 0.0
            if model_path:
                self.load(model_path)

        def score(self, sequence: np.ndarray) -> dict[str, Any]:
            if self.model is None:
                return {"lstm_score": 0.0, "reconstruction_error": 0.0, "is_anomaly": False}
            self.model.eval()
            with torch.no_grad():
                seq = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
                recon = self.model(seq)
                err = float(torch.mean((seq - recon) ** 2).cpu().numpy())
            score = min(err / (3 * self.threshold + 1e-8), 1.0)
            return {
                "lstm_score": score,
                "reconstruction_error": err,
                "is_anomaly": err > self.threshold,
            }

        def save(self, path: str) -> None:
            if self.model is None:
                raise RuntimeError("No model available to save")
            torch.save({
                "model_state": self.model.state_dict(),
                "scaler": self.scaler,
                "threshold": self.threshold,
            }, path)

        def load(self, path: str) -> None:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = LSTMAutoencoder().to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.scaler = checkpoint.get("scaler")
            self.threshold = checkpoint.get("threshold", 0.0)
else:
    class LSTMEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x


    class LSTMDecoder:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, z):
            return z


    class LSTMAutoencoder:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x

        def reconstruction_error(self, x):
            return 0.0


    class LSTMAnomalyDetector:
        def __init__(self, model_path: Optional[str] = None):
            self.available = False
            self.model = None
            self.scaler = None
            self.threshold = 0.0

        def score(self, sequence: np.ndarray) -> dict[str, Any]:
            return {"lstm_score": 0.0, "reconstruction_error": 0.0, "is_anomaly": False}

        def save(self, path: str) -> None:
            raise RuntimeError("PyTorch is not installed")

        def load(self, path: str) -> None:
            raise RuntimeError("PyTorch is not installed")
