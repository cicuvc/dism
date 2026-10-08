"""Stable optimizer topology metadata for exact checkpoint resume."""

import hashlib
import json


def optimizer_schema(model, optimizer):
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    seen = set()
    groups = []
    for group in optimizer.param_groups:
        parameter_names = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id not in names_by_id:
                raise ValueError("Optimizer contains a parameter that is not in the model")
            if parameter_id in seen:
                raise ValueError(f"Optimizer parameter appears more than once: {names_by_id[parameter_id]}")
            seen.add(parameter_id)
            parameter_names.append(names_by_id[parameter_id])
        groups.append({
            "kind": group.get("kind", type(optimizer).__name__),
            "parameters": parameter_names,
        })
    missing = [name for parameter_id, name in names_by_id.items() if parameter_id not in seen]
    if missing:
        raise ValueError(f"Optimizer is missing model parameters: {missing}")
    return {
        "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": groups,
    }


def optimizer_schema_fingerprint(schema):
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
