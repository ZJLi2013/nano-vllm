import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3Config

from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

# Alias for consistency
Qwen3MoEAttention = Qwen3Attention


class Qwen3MoEMLP(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.total_num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        # moe_experts_to_load is attached to the hf_config in the engine
        self.num_loaded_experts = getattr(config, 'moe_experts_to_load', self.total_num_experts)

        self.router = nn.Linear(self.hidden_size, self.total_num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    hidden_act=config.hidden_act,
                )
                for _ in range(self.num_loaded_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        router_logits = self.router(hidden_states_flat)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=1, dtype=torch.float), self.top_k, dim=-1
        )

        expert_mask = selected_experts < self.num_loaded_experts
        
        routing_weights = routing_weights * expert_mask
        routing_weights_sum = routing_weights.sum(dim=-1, keepdim=True)
        routing_weights_sum[routing_weights_sum == 0] = 1.0
        routing_weights = routing_weights / routing_weights_sum
        
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in range(self.num_loaded_experts):
            expert_layer = self.experts[expert_idx]
            token_indices, topk_ids = torch.where(selected_experts == expert_idx)

            if token_indices.numel() == 0:
                continue

            current_routing_weights = routing_weights[token_indices, topk_ids, None]
            current_hidden_states = hidden_states_flat[token_indices]

            expert_output = expert_layer(current_hidden_states)
            weighted_output = expert_output * current_routing_weights
            final_hidden_states.index_add_(0, token_indices, weighted_output)

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_size)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MoEMLP(config)
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
            [Qwen3MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        # The following mapping is for the experts, but the loader
        # may need to be adapted to handle the expert indexing.
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
