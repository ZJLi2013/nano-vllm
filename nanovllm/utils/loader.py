import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_model(model: nn.Module, model_path: str):
    """
    Load model weights from safetensors checkpoint files.

    Args:
        model: The model instance to load weights into
        model_path: Path to directory containing model checkpoint files
    """
    logger.info(f"Loading model weights from: {model_path}")

    # Get the number of loaded experts from the model config, if it exists
    num_loaded_experts = -1
    if (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
        and len(model.model.layers) > 0
    ):
        if hasattr(model.model.layers[0], "mlp") and hasattr(
            model.model.layers[0].mlp, "num_loaded_experts"
        ):
            num_loaded_experts = model.model.layers[0].mlp.num_loaded_experts
            logger.info(f"Loading {num_loaded_experts} experts per MoE layer")

    # Find safetensors checkpoint files
    checkpoint_files = _find_safetensors_files(model_path)

    if not checkpoint_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    logger.info(f"Found {len(checkpoint_files)} safetensors files")

    total_loaded = 0
    total_skipped = 0

    for filepath in tqdm(checkpoint_files, desc="Loading weights"):
        logger.debug(f"Loading weights from: {filepath}")

        try:
            state_dict = _load_safetensors_file(filepath)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            continue

        for weight_name, weight_data in state_dict.items():
            # Skip expert weights if we're not loading all experts
            if num_loaded_experts != -1:
                match = re.search(r"experts\.([0-9]+)\.", weight_name)
                if match:
                    expert_idx = int(match.group(1))
                    if expert_idx >= num_loaded_experts:
                        total_skipped += 1
                        continue

            # Map weight name to model parameter name
            mapped_name, shard_id = _map_weight_name(
                model, weight_name, num_loaded_experts
            )

            if mapped_name is None:
                total_skipped += 1
                continue

            # Skip layers that don't exist
            if _should_skip_layer(model, mapped_name):
                total_skipped += 1
                continue

            # Load the weight
            if _load_weight(model, mapped_name, weight_data, shard_id):
                total_loaded += 1
            else:
                total_skipped += 1

    logger.info(
        f"Weight loading complete: {total_loaded} loaded, {total_skipped} skipped"
    )


def _find_safetensors_files(model_path: str) -> list[Path]:
    """Find safetensors checkpoint files."""
    safetensors_paths = sorted(list(Path(model_path).glob("*.safetensors")))

    if not safetensors_paths:
        return []

    # Prioritize consolidated files
    consolidated = [
        p
        for p in safetensors_paths
        if "consolidated" in p.name or "pytorch_model" in p.name
    ]
    if consolidated:
        return consolidated

    return safetensors_paths


def _load_safetensors_file(filepath: Path) -> Dict[str, torch.Tensor]:
    """Load a safetensors file."""
    try:
        from safetensors import safe_open

        state_dict = {}
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    except ImportError:
        raise ImportError("safetensors package is required but not installed")
    except Exception as e:
        logger.error(f"Failed to load safetensors file {filepath}: {e}")
        raise


def _map_weight_name(
    model: nn.Module, weight_name: str, num_loaded_experts: int
) -> tuple[Optional[str], Optional[str]]:
    """
    Map checkpoint weight name to model parameter name.

    Returns:
        Tuple of (mapped_name, shard_id) or (None, None) if should be skipped
    """
    # Handle expert weights first
    is_expert_weight = False
    expert_id = None

    # Check if this is an expert weight
    match = re.search(r"experts\.([0-9]+)\.", weight_name)
    if match:
        expert_id = int(match.group(1))
        is_expert_weight = True

        # Check if we should skip this expert
        if num_loaded_experts != -1 and expert_id >= num_loaded_experts:
            return None, None

    # Try to map the weight using packed_modules_mapping
    name = weight_name
    shard_id = None

    if hasattr(model, "packed_modules_mapping"):
        for packed_name, (
            saved_name,
            shard_info,
        ) in model.packed_modules_mapping.items():
            if saved_name not in weight_name:
                continue

            # Handle expert weights
            if is_expert_weight and "experts" in saved_name:
                # Build the target parameter name
                name = weight_name.replace(
                    f"experts.{expert_id}.{saved_name}", f"{packed_name}"
                )
                # Set shard_id for expert weight loading
                shard_id = f"expert_{expert_id}"
                break
            elif not is_expert_weight:
                # Handle regular weights
                if isinstance(shard_info, int):
                    name = weight_name.replace(
                        saved_name, f"{packed_name}.{shard_info}"
                    )
                    shard_id = shard_info
                else:
                    name = weight_name.replace(
                        saved_name, f"{packed_name}.{shard_info}"
                    )
                    shard_id = shard_info
                break

    return name, shard_id


def _should_skip_layer(model: nn.Module, param_name: str) -> bool:
    """Check if a parameter should be skipped because the layer doesn't exist."""
    match = re.match(r"model\.layers\.(\d+)\.", param_name)
    if match:
        layer_idx = int(match.group(1))
        if (
            hasattr(model, "model")
            and hasattr(model.model, "layers")
            and isinstance(model.model.layers, nn.ModuleList)
        ):
            if layer_idx >= len(model.model.layers):
                logger.debug(
                    f"Skipping parameter {param_name} - layer {layer_idx} doesn't exist"
                )
                return True
    return False


def _load_weight(
    model: nn.Module, param_name: str, weight_data: torch.Tensor, shard_id: Any
) -> bool:
    """Load a single weight into the model."""
    try:
        param = model.get_parameter(param_name)

        # Handle weight splitting for tensor parallelism
        if param.shape != weight_data.shape:
            weight_data = _split_weight_for_tp(param_name, weight_data, param.shape)
            if weight_data is None:
                logger.warning(
                    f"Shape mismatch for {param_name}: expected {param.shape}, got {weight_data.shape}"
                )
                return False

        # Use custom weight loader if available
        if hasattr(param, "weight_loader") and shard_id is not None:
            param.weight_loader(param, weight_data, shard_id)
        else:
            param.data.copy_(weight_data)

        return True

    except AttributeError:
        logger.debug(f"Parameter not found: {param_name}")
        return False


def _split_weight_for_tp(
    param_name: str, weight_data: torch.Tensor, target_shape: torch.Size
) -> Optional[torch.Tensor]:
    """Split weights for tensor parallelism."""
    if "qkv_proj" in param_name:
        wq, wk, wv = torch.chunk(weight_data, 3, dim=0)
        if "qkv_proj.q" in param_name:
            return wq
        elif "qkv_proj.k" in param_name:
            return wk
        elif "qkv_proj.v" in param_name:
            return wv
    elif "gate_up_proj" in param_name:
        w_gate, w_up = torch.chunk(weight_data, 2, dim=0)
        if "gate_up_proj.0" in param_name:
            return w_gate
        elif "gate_up_proj.1" in param_name:
            return w_up

    return None
