#!/usr/bin/env python3
"""
Test script to debug weight loading for Qwen3 MoE model
"""

import torch
import torch.distributed as dist
import os
import argparse
from transformers import Qwen3Config, AutoConfig

from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.utils.loader import load_model


def setup_distributed():
    """Initialize distributed process group for single GPU testing"""
    if not dist.is_initialized():
        # Initialize with single process for testing
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_weight_loading(model_path):
    """Test weight loading from a real model path"""
    print(f"Testing Qwen3 MoE weight loading from: {model_path}")

    # Setup distributed environment for testing
    setup_distributed()

    # Load config from the model path
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_path)

    # Ensure it's a Qwen3 MoE config
    if not hasattr(config, "num_experts") or config.num_experts <= 1:
        print("⚠️  Warning: This doesn't appear to be a MoE model (num_experts <= 1)")

    print(
        f"Model config: {config.num_hidden_layers} layers, {config.num_experts} experts"
    )

    # Initialize model
    model = Qwen3MoeForCausalLM(config)

    print(f"Model created with {config.num_experts} experts")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Try to load weights from the real model path
    print("Attempting to load weights from model directory...")
    try:
        load_model(model, model_path)
        print("✅ Weight loading successful!")

        # Test forward pass
        print("Testing forward pass...")
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        positions = torch.tensor([[0, 1, 2, 3, 4]])

        with torch.no_grad():
            output = model(input_ids, positions)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Error during weight loading: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up distributed environment
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", required=True, help="Path to the model directory"
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    test_weight_loading(model_path)


if __name__ == "__main__":
    main()
