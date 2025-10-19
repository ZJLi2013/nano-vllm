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

    # Create expert mapping lookup for faster access
    expert_lookup = {
        weight_name: (param_name, expert_id, shard_id)
        for param_name, weight_name, expert_id, shard_id in expert_mapping
    }

    # Find all safetensors files
    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {path}")

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)

                # 1. Try to load as an MoE expert weight
                if weight_name in expert_lookup:
                    param_name, expert_id, shard_id = expert_lookup[weight_name]
                    try:
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(
                            param,
                            loaded_weight,
                            expert_id=expert_id,
                            shard_id=shard_id,
                        )
                    except AttributeError:
                        # This can happen if the expert is not loaded (subset of experts)
                        print(f"Skipping expert weight: {weight_name}")
                    continue  # Move to the next weight

                # 2. Try to load as a dense packed-layer weight
                dense_packed_match = None
                if "experts" not in weight_name:
                    for k, (v, shard_info) in packed_modules_mapping.items():
                        if k in weight_name:
                            dense_packed_match = (k, v, shard_info)
                            break
                
                if dense_packed_match:
                    k, v, shard_info = dense_packed_match
                    try:
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, shard_info)
                    except AttributeError:
                        print(f"Skipping packed weight: {weight_name}")
                    continue # Move to the next weight

                # 3. Try to load as a default weight
                try:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                except AttributeError:
                    print(f"Skipping weight: {weight_name}")
