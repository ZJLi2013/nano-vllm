import re
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm


def load_model(model: nn.Module, model_path: str):
    # Get the number of loaded experts from the model config, if it exists
    num_loaded_experts = -1
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        if hasattr(model.model.layers[0], "mlp") and hasattr(model.model.layers[0].mlp, "num_loaded_experts"):
            num_loaded_experts = model.model.layers[0].mlp.num_loaded_experts

    paths = sorted(list(Path(model_path).glob("*.safetensors")))
    if not paths:
        paths = sorted(list(Path(model_path).glob("*.bin")))
    unsharded_path = []
    for p in paths:
        if "consolidated" in p.name or "pytorch_model" in p.name:
            unsharded_path.append(p)

    for filepath in tqdm(unsharded_path):
        state_dict = torch.load(filepath, map_location="cpu", mmap=True)
        for weight_name, weight_data in state_dict.items():
            # Check if the weight belongs to an expert we are not loading
            if num_loaded_experts != -1:
                match = re.search(r'experts\.([0-9]+)\.', weight_name)
                if match:
                    expert_idx = int(match.group(1))
                    if expert_idx >= num_loaded_experts:
                        continue  # Skip this weight

            for packed_name, (
                saved_name,
                shard_id,
            ) in model.packed_modules_mapping.items():
                if saved_name not in weight_name:
                    continue
                if isinstance(shard_id, int):
                    name = weight_name.replace(saved_name, f"{packed_name}.{shard_id}")
                else:
                    name = weight_name.replace(saved_name, f"{packed_name}.{shard_id}")
                break
            else:
                name = weight_name

            match = re.match(r"model\.layers\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                if (
                    hasattr(model, "model")
                    and hasattr(model.model, "layers")
                    and isinstance(model.model.layers, nn.ModuleList)
                ):
                    if layer_idx >= len(model.model.layers):
                        continue

            param = model.get_parameter(name)
            if param.shape != weight_data.shape:
                if "qkv_proj" in name:
                    wq, wk, wv = torch.chunk(weight_data, 3, dim=0)
                    if "qkv_proj.q" in name:
                        weight_data = wq
                    elif "qkv_proj.k" in name:
                        weight_data = wk
                    elif "qkv_proj.v" in name:
                        weight_data = wv
                elif "gate_up_proj" in name:
                    w_gate, w_up = torch.chunk(weight_data, 2, dim=0)
                    if "gate_up_proj.0" in name:
                        weight_data = w_gate
                    elif "gate_up_proj.1" in name:
                        weight_data = w_up
            param.copy_(weight_data)
