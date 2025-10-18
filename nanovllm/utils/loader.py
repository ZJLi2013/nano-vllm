import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    expert_mapping = getattr(model, "get_expert_mapping", lambda: [])()

    # Create expert mapping lookup
    expert_lookup = {}
    for param_name, weight_name, expert_id, shard_id in expert_mapping:
        expert_lookup[weight_name] = (param_name, expert_id, shard_id)

    # Find all safetensors files
    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {path}")

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                is_packed = False

                # Check for expert mapping first
                if weight_name in expert_lookup:
                    param_name, expert_id, shard_id = expert_lookup[weight_name]
                    try:
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight, f"expert_{expert_id}")
                        is_packed = True
                    except AttributeError:
                        print(f"Skipping expert weight: {weight_name}")
                        continue

                # Check for packed modules
                if not is_packed:
                    for k, (v, shard_info) in packed_modules_mapping.items():
                        if k in weight_name:
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, loaded_weight, shard_info)
                            is_packed = True
                            break

                # Default loading for non-packed weights
                if not is_packed:
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    except AttributeError:
                        print(f"Skipping weight: {weight_name}")
