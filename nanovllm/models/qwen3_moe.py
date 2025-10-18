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

        # Use single large tensors to store all expert weights
        # gate_up_weights: [num_experts, 2 * intermediate_size, hidden_size]
        # down_weights: [num_experts, hidden_size, intermediate_size]
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
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        """Load gate_up expert weights"""
        # loaded_weight: [2 * intermediate_size, hidden_size]
        # loaded_shard_id format: "expert_{expert_id}"
        if loaded_shard_id.startswith("expert_"):
            expert_id = int(loaded_shard_id.split("_")[1])
            # Split by TP
            shard_size = loaded_weight.size(0) // self.tp_size
            start_idx = self.tp_rank * shard_size
            weight_shard = loaded_weight.narrow(0, start_idx, shard_size)
            param.data[expert_id].copy_(weight_shard)

    def down_weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        """Load down expert weights"""
        # loaded_weight: [hidden_size, intermediate_size]
        # loaded_shard_id format: "expert_{expert_id}"
        if loaded_shard_id.startswith("expert_"):
            expert_id = int(loaded_shard_id.split("_")[1])
            # Split columns by TP
            shard_size = loaded_weight.size(1) // self.tp_size
            start_idx = self.tp_rank * shard_size
            weight_shard = loaded_weight.narrow(1, start_idx, shard_size)
            param.data[expert_id].copy_(weight_shard)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D input shapes
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_dim)
            original_shape = (batch_size, seq_len, hidden_dim)
        else:
            # Already flattened: [num_tokens, hidden_dim]
            hidden_states_flat = hidden_states
            original_shape = hidden_states.shape

        # 1. Routing computation
        router_logits = self.gate(hidden_states_flat)  # [num_tokens, num_experts]

        # 2. Top-k expert selection
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float), self.top_k, dim=-1
        )  # [num_tokens, top_k]

        # 3. Probability normalization (if configured)
        if self.norm_topk_prob:
            routing_weights_sum = routing_weights.sum(dim=-1, keepdim=True)
            routing_weights_sum[routing_weights_sum == 0] = 1.0
            routing_weights = routing_weights / routing_weights_sum

        routing_weights = routing_weights.to(hidden_states.dtype)

        # 4. Expert computation
        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Compute expert outputs for each token
        for token_idx in range(hidden_states_flat.size(0)):
            token_hidden = hidden_states_flat[token_idx]  # [hidden_dim]
            token_experts = selected_experts[token_idx]  # [top_k]
            token_weights = routing_weights[token_idx]  # [top_k]

            token_output = torch.zeros_like(token_hidden)

            for expert_idx, weight in zip(token_experts, token_weights):
                if expert_idx >= self.total_num_experts:
                    continue

                # Compute expert output - using nano-vllm's parallel linear layers
                # gate_up computation
                gate_up_output = F.linear(
                    token_hidden,
                    self.gate_up_weights[expert_idx],  # [2*inter_size/tp, hidden_dim]
                )  # [2*inter_size/tp]

                # Use SiluAndMul activation function
                activated = self.act_fn(gate_up_output)  # [inter_size/tp]

                # down projection
                expert_output = F.linear(
                    activated,
                    self.down_weights[expert_idx],  # [hidden_dim, inter_size/tp]
                )  # [hidden_dim]

                token_output += expert_output * weight

            final_hidden_states[token_idx] = token_output

        # 5. TP all_gather
        if self.tp_size > 1:
            # Gather results across all TP ranks
            gathered_outputs = [
                torch.zeros_like(final_hidden_states) for _ in range(self.tp_size)
            ]
            dist.all_gather(gathered_outputs, final_hidden_states)
            final_hidden_states = torch.cat(gathered_outputs, dim=0)

        # Reshape back to original shape
        if len(original_shape) == 3:
            return final_hidden_states.view(original_shape)
        else:
            return final_hidden_states


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
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "mlp.experts.gate_proj": ("mlp.gate_up_weights", "expert"),
        "mlp.experts.up_proj": ("mlp.gate_up_weights", "expert"),
        "mlp.experts.down_proj": ("mlp.down_weights", "expert"),
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
