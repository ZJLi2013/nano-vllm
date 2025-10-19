import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3MoEAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MoEMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.total_num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)

        # TP configuration
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # gate_up_weights: [num_experts, 2 * intermediate_size/tp, hidden_size]
        # down_weights: [num_experts, hidden_size, intermediate_size/tp]
        self.gate_up_weights = nn.Parameter(
            torch.empty(
                self.total_num_experts,
                2 * self.intermediate_size // self.tp_size,
                self.hidden_size,
            )
        )
        self.down_weights = nn.Parameter(
            torch.empty(
                self.total_num_experts,
                self.hidden_size,
                self.intermediate_size // self.tp_size,
            )
        )

        # Set weight loaders
        self.gate_up_weights.weight_loader = self.gate_up_weight_loader
        self.down_weights.weight_loader = self.down_weight_loader

        # Gate layer (replicated across all TP ranks)
        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.total_num_experts,
            bias=False,
        )

        # Activation function
        self.act_fn = SiluAndMul()

    def gate_up_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ):
        """Load gate_up expert weights with proper TP and shard handling."""
        tp_rank = self.tp_rank
        tp_size = self.tp_size
        inter_size = self.intermediate_size

        # This is a MergedColumnParallelLinear, sharded on the output dimension (dim 0).
        # The full weight for gate_proj or up_proj is [intermediate_size, hidden_size].
        shard_size = inter_size // tp_size
        start_row = tp_rank * shard_size
        weight_shard = loaded_weight.narrow(0, start_row, shard_size)

        # Copy into the correct slice of the expert's parameter.
        if shard_id == "gate":
            # First half of the rows in the expert's gate_up weight
            param_slice = param.data[expert_id].narrow(0, 0, shard_size)
        elif shard_id == "up":
            # Second half of the rows
            param_slice = param.data[expert_id].narrow(0, shard_size, shard_size)
        else:
            raise ValueError(f"Invalid shard_id {shard_id} for gate_up_weight_loader")
        param_slice.copy_(weight_shard)

    def down_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ):
        """Load down expert weights with proper TP handling."""
        tp_rank = self.tp_rank
        tp_size = self.tp_size
        inter_size = self.intermediate_size

        # This is a RowParallelLinear, sharded on the input dimension (dim 1).
        # The full weight is [hidden_size, intermediate_size].
        shard_size = inter_size // tp_size
        start_col = tp_rank * shard_size
        weight_shard = loaded_weight.narrow(1, start_col, shard_size)

        # The parameter is already sharded for this TP rank, so we can copy directly.
        param.data[expert_id].copy_(weight_shard)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Reshape and Router Logits
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)

        # 2. Top-k Selection
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float),
            self.top_k,
            dim=-1,
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 3. Batched Expert Computation
        final_hidden_states = self.compute_batched_experts(
            hidden_states, routing_weights, selected_experts
        )

        # 4. All-to-All for Tensor Parallelism
        if self.tp_size > 1:
            final_hidden_states = self.all_to_all_comm(final_hidden_states)

        return final_hidden_states.view(original_shape)

    def compute_batched_experts(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Computes expert outputs in a batched manner."""
        num_tokens, _ = hidden_states.shape
        final_hidden_states = torch.zeros_like(hidden_states)

        # Create a mask for valid expert selections
        expert_mask = selected_experts < self.total_num_experts
        # Flatten the expert indices and routing weights for easier processing
        flat_expert_indices = selected_experts.flatten()
        flat_routing_weights = routing_weights.flatten()
        # Create a flat list of token indices, repeated for each expert choice
        flat_token_indices = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .flatten()
        )

        # Group tokens by the expert they are assigned to
        for expert_id in range(self.total_num_experts):
            # Find which tokens are routed to this expert
            token_mask = (flat_expert_indices == expert_id) & expert_mask.flatten()
            if not torch.any(token_mask):
                continue

            # Get the indices and hidden states of the tokens for this expert
            expert_token_indices = flat_token_indices[token_mask]
            expert_hidden_states = hidden_states[expert_token_indices]

            # Perform the expert computation in a single batch
            gate_up_output = F.linear(
                expert_hidden_states, self.gate_up_weights[expert_id]
            )
            activated_output = self.act_fn(gate_up_output)
            down_output = F.linear(activated_output, self.down_weights[expert_id])

            # Apply the routing weights
            weighted_output = (
                down_output * flat_routing_weights[token_mask].unsqueeze(1)
            )

            # Add the expert's output to the final hidden states
            final_hidden_states.index_add_(0, expert_token_indices, weighted_output)

        return final_hidden_states

    def all_to_all_comm(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Handles all-to-all communication for tensor parallelism."""
        # This is a placeholder for the actual all-to-all implementation.
        # In a real TP setup, you would split the output and exchange parts
        # with other ranks. For now, we'll simulate the communication overhead
        # with a simple all-gather, which is less efficient but functionally similar
        # for a single-node setup.
        if self.tp_size > 1:
            gathered_outputs = [
                torch.zeros_like(expert_outputs) for _ in range(self.tp_size)
            ]
            dist.all_gather(gathered_outputs, expert_outputs)
            # In a true all-to-all, each rank would receive a different part of the data.
            # Here, we just sum them up, which is not correct for a real model,
            # but serves as a placeholder to ensure the dimensions are correct.
            # A correct implementation would involve splitting the hidden dimension
            # and scattering/gathering the appropriate chunks.
            return torch.sum(torch.stack(gathered_outputs), dim=0)
        return expert_outputs


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # Determine whether to use MoE or regular MLP based on decoder_sparse_step
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        mlp_only_layers = getattr(config, "mlp_only_layers", [])

        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        # Maps checkpoint weight name part to (model parameter name, shard_id)
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Get expert weight mapping for MoE layers.

        Returns:
            List of tuples: (param_name, weight_name, expert_id, shard_id)
        """
        expert_mapping = []
        expert_weights = [
            ("gate_proj", "gate_up_weights", "gate"),
            ("up_proj", "gate_up_weights", "up"),
            ("down_proj", "down_weights", "down"),
        ]

        for layer_idx, layer in enumerate(self.model.layers):
            # Check if this layer has an MLP block with experts
            if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "total_num_experts")):
                continue

            # If it's an MoE layer, get the total number of experts from it
            if layer.mlp.total_num_experts > 0:
                num_experts_in_layer = layer.mlp.total_num_experts

                for expert_id in range(num_experts_in_layer):
                    for ckpt_name, param_name_suffix, shard_id in expert_weights:
                        param_name = f"model.layers.{layer_idx}.mlp.{param_name_suffix}"
                        weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{ckpt_name}.weight"
                        expert_mapping.append(
                            (param_name, weight_name, expert_id, shard_id)
                        )

        return expert_mapping
