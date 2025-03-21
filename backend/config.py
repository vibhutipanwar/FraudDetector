"""
Configuration settings for the Fraud Detection, Alert, and Monitoring (FDAM) system.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENVIRONMENT", "development")

# Default config values
DEFAULT_CONFIG = {
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": True if ENV != "production" else False,
        "workers": 4,
        "response_time_threshold_ms": 300,
    },
    "database": {
        "mongodb": {
            "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
            "db_name": os.getenv("MONGODB_DB", "fraud_detection"),
            "collections": {
                "transactions": "transactions",
                "fraud_reports": "fraud_reports",
                "rules": "rules",
                "model_metrics": "model_metrics",
            },
            "connection_pool_size": 10,
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "db": int(os.getenv("REDIS_DB", 0)),
            "cache_ttl": 300,  # 5 minutes
            "fraud_results_ttl": 3600,  # 1 hour
        },
    },
    "ml": {
        "model_path": os.path.join(BASE_DIR, "models"),
        "default_model": "xgboost_fraud_detector_v1.pkl",
        "backup_model": "random_forest_fraud_detector_v1.pkl",
        "anomaly_model": "isolation_forest_v1.pkl",
        "confidence_threshold": 0.8,
        "retrain_frequency_days": 7,
    },
    "rule_engine": {
        "rules_config_path": os.path.join(BASE_DIR, "config", "rules.yaml"),
        "max_rules_per_category": 25,
        "default_rules_enabled": True,
    },
    "monitoring": {
        "log_level": "INFO" if ENV == "production" else "DEBUG",
        "performance_tracking": True,
        "metrics_interval_seconds": 60,
    },
    "security": {
        "api_key_header": "X-API-Key",
        "jwt_secret": os.getenv("JWT_SECRET", "dev-secret-key-change-in-production"),
        "jwt_algorithm": "HS256",
        "jwt_expiration_minutes": 60,
    },
}


class Config:
    """Configuration management for the application."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from environment and files."""
        self._config = DEFAULT_CONFIG.copy()

        # Load environment-specific config if available
        env_config_path = os.path.join(BASE_DIR, "config", f"{ENV}.yaml")
        if os.path.exists(env_config_path):
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f)
                self._deep_update(self._config, env_config)

        # Override with environment variables
        # Example: FDAM_API_PORT=8080 would override api.port
        for key, value in os.environ.items():
            if key.startswith("FDAM_"):
                parts = key[5:].lower().split("_")
                self._update_nested_dict(self._config, parts, value)

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]):
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def _update_nested_dict(self, d: Dict[str, Any], keys: list, value: Any):
        """Update a nested dictionary using a list of keys."""
        if len(keys) == 1:
            try:
                # Try to convert string to appropriate type
                if isinstance(d.get(keys[0]), bool):
                    d[keys[0]] = value.lower() in ("true", "yes", "1")
                elif isinstance(d.get(keys[0]), int):
                    d[keys[0]] = int(value)
                elif isinstance(d.get(keys[0]), float):
                    d[keys[0]] = float(value)
                else:
                    d[keys[0]] = value
            except (ValueError, AttributeError):
                d[keys[0]] = value
        elif len(keys) > 1 and keys[0] in d:
            if not isinstance(d[keys[0]], dict):
                d[keys[0]] = {}
            self._update_nested_dict(d[keys[0]], keys[1:], value)

    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Example: config.get("api.port")
        """
        keys = key_path.split(".")
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    def set(self, key_path: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Example: config.set("api.port", 8080)
        """
        keys = key_path.split(".")
        self._update_nested_dict(self._config, keys, value)

    @property
    def all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config.copy()


# Create a singleton instance
config = Config()
