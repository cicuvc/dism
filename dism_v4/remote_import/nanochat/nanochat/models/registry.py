"""Registration and construction of model architectures."""

from dataclasses import dataclass
from typing import Callable

from nanochat.models.spec import ModelSpec


@dataclass(frozen=True)
class ModelRegistration:
    architecture: str
    architecture_version: int
    config_cls: type
    model_cls: type
    capabilities: object
    scaling_reference_spec: Callable[[ModelSpec], ModelSpec] | None = None
    migrate_legacy_config: Callable[[dict], dict] | None = None
    migrate_state_dict: Callable[[dict, object], dict] | None = None


_REGISTRY = {}


def register_model(
    architecture,
    architecture_version,
    config_cls,
    model_cls,
    capabilities,
    *,
    scaling_reference_spec=None,
    migrate_legacy_config=None,
    migrate_state_dict=None,
):
    if architecture in _REGISTRY:
        raise ValueError(f"Model architecture is already registered: {architecture}")
    _REGISTRY[architecture] = ModelRegistration(
        architecture=architecture,
        architecture_version=architecture_version,
        config_cls=config_cls,
        model_cls=model_cls,
        capabilities=capabilities,
        scaling_reference_spec=scaling_reference_spec,
        migrate_legacy_config=migrate_legacy_config,
        migrate_state_dict=migrate_state_dict,
    )


def get_model_registration(architecture):
    try:
        return _REGISTRY[architecture]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown model architecture {architecture!r}; registered: {available}"
        ) from exc


def build_model(spec):
    registration = get_model_registration(spec.architecture)
    if spec.architecture_version != registration.architecture_version:
        raise ValueError(
            f"Unsupported {spec.architecture} version {spec.architecture_version}; "
            f"runtime supports version {registration.architecture_version}"
        )
    config = registration.config_cls(**spec.config)
    model = registration.model_cls(config)
    # The spec is framework metadata rather than part of the torch state dict.
    model.model_spec = spec
    model.capabilities = registration.capabilities
    return model


def get_scaling_reference_spec(spec):
    registration = get_model_registration(spec.architecture)
    if registration.scaling_reference_spec is None:
        raise ValueError(
            f"Architecture {spec.architecture!r} does not define a scaling reference model"
        )
    reference = registration.scaling_reference_spec(spec)
    if reference.architecture != spec.architecture:
        raise ValueError("Scaling reference must use the same architecture")
    return reference


def model_spec_from_metadata(meta_data):
    """Read a current ModelSpec or upgrade a pre-registry GPT checkpoint."""
    if "model_spec" in meta_data:
        spec = ModelSpec.from_dict(meta_data["model_spec"])
        expected = meta_data.get("model_fingerprint")
        if expected is not None and expected != spec.fingerprint():
            raise ValueError("Checkpoint model fingerprint does not match model_spec")
        # Accessing the registration here fails early for unavailable models.
        get_model_registration(spec.architecture)
        return spec

    registration = get_model_registration("nanochat_v1")
    config = dict(meta_data["model_config"])
    if registration.migrate_legacy_config is not None:
        config = registration.migrate_legacy_config(config)
    return ModelSpec(
        architecture=registration.architecture,
        architecture_version=registration.architecture_version,
        config=config,
    )


def migrate_model_state_dict(spec, model_data, model_config):
    registration = get_model_registration(spec.architecture)
    if registration.migrate_state_dict is None:
        return model_data
    return registration.migrate_state_dict(model_data, model_config)
