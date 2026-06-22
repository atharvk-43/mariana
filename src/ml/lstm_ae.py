import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.threshold = 0.0
        if model_path:
            self.load(model_path)

    def score(self, sequence: np.ndarray) -> dict:
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
        torch.save({
            "model_state": self.model.state_dict(),
            "model_params": {"n_features": self.model.encoder.lstm.input_size, "hidden_dim": self.model.encoder.lstm.hidden_size, "seq_len": self.model.decoder.seq_len},
            "scaler": self.scaler,
            "threshold": self.threshold,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        params = checkpoint.get("model_params", {})
        self.model = LSTMAutoencoder(
            n_features=params.get("n_features", 23),
            hidden_dim=params.get("hidden_dim", 256),
            seq_len=params.get("seq_len", 60),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler = checkpoint.get("scaler")
        self.threshold = checkpoint.get("threshold", 0.0)
