"""Strict loading for current and original (block-18) LiSalNet checkpoints.

Only the six known entries of the removed block may be discarded. All active
parameters and buffers must be present with matching shapes. No strict=False
fallback is used. Legacy optimizer state is not migrated by this module.
"""
from collections import OrderedDict
from collections.abc import Mapping


REMOVED_SHAPES = {
    "Backbone.features.18.0.weight": (1280, 320, 1, 1),
    "Backbone.features.18.1.weight": (1280,),
    "Backbone.features.18.1.bias": (1280,),
    "Backbone.features.18.1.running_mean": (1280,),
    "Backbone.features.18.1.running_var": (1280,),
    "Backbone.features.18.1.num_batches_tracked": (),
}


def read_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a state dict or a checkpoint containing state_dict.")
    for name in ("state_dict", "model_state_dict"):
        if name in checkpoint:
            checkpoint = checkpoint[name]
            break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("Checkpoint contains no model state dictionary.")
    normalized = OrderedDict()
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not hasattr(value, "shape"):
            raise TypeError("Model state must map string keys to tensors.")
        name = key[len("module."):] if key.startswith("module.") else key
        if name in normalized:
            raise ValueError(f"Duplicate state key after removing module. prefix: {name}")
        normalized[name] = value
    return normalized


def prepare_state_dict(checkpoint, expected_state):
    state = extract_state_dict(checkpoint)
    removed = []
    for key, shape in REMOVED_SHAPES.items():
        if key in state:
            if tuple(state[key].shape) != shape:
                raise ValueError(f"Unexpected legacy block shape: {key}: {tuple(state[key].shape)}")
            removed.append(key)
            del state[key]
    missing = sorted(set(expected_state) - set(state))
    unexpected = sorted(set(state) - set(expected_state))
    mismatched = {
        key: {"checkpoint": tuple(state[key].shape), "model": tuple(expected_state[key].shape)}
        for key in set(state) & set(expected_state)
        if tuple(state[key].shape) != tuple(expected_state[key].shape)
    }
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Checkpoint does not match LiSalNet. Missing={missing}; "
            f"unexpected={unexpected}; shape_mismatches={mismatched}"
        )
    return state, removed


def load_model_state(model, checkpoint):
    state, removed = prepare_state_dict(checkpoint, model.state_dict())
    model.load_state_dict(state, strict=True)
    if removed:
        print("Discarded only the removed RGB block entries: " + ", ".join(removed))
    return removed


def load_checkpoint(model, path):
    return load_model_state(model, read_checkpoint(path))
