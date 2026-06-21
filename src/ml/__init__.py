from .features import build_feature_matrix, get_feature_vector_for_if, get_sequence_for_lstm, get_graph_features, normalize_features
from .isolation_forest import IsolationForestDetector
from .lstm_ae import LSTMAutoencoder, LSTMAnomalyDetector
from .gat_model import GATAutoencoder, GATAnomalyDetector
from .ensemble import EnsembleDetector

try:
    from ml.prophet_model import ProphetForecaster, PROPHET_METRICS
    _prophet_ok = True
except ImportError:
    ProphetForecaster = None
    PROPHET_METRICS = []
    _prophet_ok = False

__all__ = [
    "build_feature_matrix", "get_feature_vector_for_if", "get_sequence_for_lstm",
    "get_graph_features", "normalize_features",
    "IsolationForestDetector",
    "LSTMAutoencoder", "LSTMAnomalyDetector",
    "ProphetForecaster", "PROPHET_METRICS",
    "GATAutoencoder", "GATAnomalyDetector",
    "EnsembleDetector",
]
