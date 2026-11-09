"""Convert Google's Keras dense identity decoder to a TensorFlow-free NumPy asset."""

from pathlib import Path
import json
import h5py
import numpy as np

root = Path(__file__).resolve().parents[1]
with h5py.File(root / "models/identity_decoder_model.h5") as model:
    config = json.loads(model.attrs["model_config"])["config"]
    layers = config["layers"]
    if any(
        layer["class_name"] not in {"InputLayer", "Concatenate", "Dense"}
        for layer in layers
    ):
        raise ValueError("Expected the public GNM dense identity decoder")
    dense = [layer for layer in layers if layer["class_name"] == "Dense"]
    arrays = {"latent_dim": np.asarray(64), "layers": np.asarray(len(dense))}
    for i, layer in enumerate(dense):
        name = layer["config"]["name"]
        weights = model["model_weights"][name][name]
        arrays[f"kernel_{i}"] = weights["kernel:0"][:]
        arrays[f"bias_{i}"] = weights["bias:0"][:]
        arrays[f"activation_{i}"] = np.asarray(layer["config"]["activation"])
np.savez_compressed(root / "models/identity_decoder.npz", **arrays)
print("GNM_SEMANTIC_DECODER_READY")
