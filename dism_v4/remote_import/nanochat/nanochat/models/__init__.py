"""Model architecture registry and common model metadata."""

from nanochat.models.registry import (
    build_model,
    get_model_registration,
    get_scaling_reference_spec,
    migrate_model_state_dict,
    model_spec_from_metadata,
    register_model,
)
from nanochat.models.spec import ModelSpec
from nanochat.models.protocol import ModelCapabilities, PretrainOptimizerConfig
from nanochat.models.optimizer import optimizer_schema, optimizer_schema_fingerprint

# Import built-in registrations after the public registry symbols exist.
from nanochat.models import nanochat_v1 as _nanochat_v1  # noqa: F401,E402
from nanochat.models import llama2 as _llama2  # noqa: F401,E402
from nanochat.models import dism as _dism  # noqa: F401,E402

__all__ = [
    "ModelSpec",
    "ModelCapabilities",
    "PretrainOptimizerConfig",
    "optimizer_schema",
    "optimizer_schema_fingerprint",
    "build_model",
    "get_model_registration",
    "get_scaling_reference_spec",
    "migrate_model_state_dict",
    "model_spec_from_metadata",
    "register_model",
]
