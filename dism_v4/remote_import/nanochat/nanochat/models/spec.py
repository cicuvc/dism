"""Serializable identity of a model architecture and its complete configuration."""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    architecture_version: int
    config: dict

    def __post_init__(self):
        if not self.architecture:
            raise ValueError("Model architecture must not be empty")
        if self.architecture_version < 1:
            raise ValueError("Model architecture_version must be positive")
        # Own a JSON-safe copy so later mutations of the caller's input do not
        # change this spec.
        canonical_config = json.loads(json.dumps(self.config, sort_keys=True))
        object.__setattr__(self, "config", canonical_config)

    @classmethod
    def from_config(cls, architecture, architecture_version, config):
        config_dict = asdict(config) if is_dataclass(config) else dict(config)
        return cls(architecture, architecture_version, config_dict)

    @classmethod
    def from_dict(cls, value):
        return cls(
            architecture=value["architecture"],
            architecture_version=value["architecture_version"],
            config=value["config"],
        )

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "architecture_version": self.architecture_version,
            "config": json.loads(json.dumps(self.config, sort_keys=True)),
        }

    def canonical_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self):
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
