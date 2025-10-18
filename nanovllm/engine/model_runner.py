import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

MODEL_REGISTRY = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        print(f"[DEBUG ModelRunner] Starting initialization for rank {rank}")
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        print(f"[DEBUG ModelRunner] Initializing process group for rank {rank}")
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        print(f"[DEBUG ModelRunner] Process group initialized for rank {rank}")
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        architecture = hf_config.architectures[0]
        model_class = MODEL_REGISTRY.get(architecture)
        if model_class is None:
            raise ValueError(f"Unsupported model architecture: {architecture}")
        print(f"[DEBUG ModelRunner] Creating model instance for rank {rank}")
        self.model = model_class(hf_config)
        print(f"[DEBUG ModelRunner] Loading model weights for rank {rank}")
        load_model(self.model, config.model)
        print(f"[DEBUG ModelRunner] Weights loaded for rank {rank}")
        self.sampler = Sampler()
        print(f"[DEBUG ModelRunner] Warming up model for rank {rank}")
        self.warmup_model()
        print(f"[DEBUG ModelRunner] Allocating KV cache for rank {rank}")
        self.allocate_kv_cache()
        if not self.enforce_eager:
            print(f"[DEBUG ModelRunner] Capturing CUDA graphs for rank {rank}")
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                print(f"[DEBUG ModelRunner] Creating shared memory for rank {rank}")
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
                print(f"[DEBUG ModelRunner] Rank {rank} passed barrier")
            else:
                dist.barrier()
                print(f"[DEBUG ModelRunner] Rank {rank} passed barrier")
                self.shm = SharedMemory(name="nanovllm")
                print(f"[DEBUG ModelRunner] Starting loop for rank {rank}")
                self.loop()
        print(f"[DEBUG ModelRunner] Initialization completed for rank {rank}")

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        print(f"[DEBUG warmup_model] Starting warmup for rank {self.rank}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        print(
            f"[DEBUG warmup_model] Creating {num_seqs} sequences of length {max_model_len}"
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        print(f"[DEBUG warmup_model] Calling run() with {len(seqs)} sequences")
        self.run(seqs, True)
        print(f"[DEBUG warmup_model] Run completed, emptying cache")
        torch.cuda.empty_cache()
        print(f"[DEBUG warmup_model] Warmup completed for rank {self.rank}")

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        if self.rank == 0:
            print("\n[DEBUG] --- Inside allocate_kv_cache ---")
            print(f"[DEBUG] Total GPU Memory: {total / 1024**3:.2f} GiB")
            print(f"[DEBUG] Free GPU Memory: {free / 1024**3:.2f} GiB")
            print(f"[DEBUG] Used GPU Memory (by everything): {used / 1024**3:.2f} GiB")
            print(
                f"[DEBUG] Peak PyTorch Memory (during load): {peak / 1024**3:.2f} GiB"
            )
            print(f"[DEBUG] Current PyTorch Memory: {current / 1024**3:.2f} GiB")
            print(
                f"[DEBUG] GPU Memory Utilization Target: {config.gpu_memory_utilization}"
            )

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )

        if self.rank == 0:
            print(
                f"[DEBUG] Size of one KV cache block: {block_bytes / 1024**2:.2f} MiB"
            )

        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )

        if self.rank == 0:
            kv_cache_size_bytes = config.num_kvcache_blocks * block_bytes
            print(f"[DEBUG] Calculated num_kvcache_blocks: {config.num_kvcache_blocks}")
            print(
                f"[DEBUG] Attempting to allocate KV cache of size: {kv_cache_size_bytes / 1024**3:.2f} GiB"
            )
            print("[DEBUG] --- End of allocate_kv_cache prints ---\n")

        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        """
            给定 block_size=1， len(seq.block_table) == seqlen
            per seq 的 block_table 都对齐到 batch 中 max_seq_len，不到长度的补 -1  
        """
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        print(f"[DEBUG prepare_prefill] Starting with {len(seqs)} sequences")
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        print(f"[DEBUG prepare_prefill] Processing sequences...")
        for seq_idx, seq in enumerate(seqs):
            seqlen = len(seq)
            print(
                f"[DEBUG prepare_prefill] Sequence {seq_idx}: length={seqlen}, num_cached_tokens={seq.num_cached_tokens}"
            )
            input_ids.extend(
                seq[seq.num_cached_tokens :]
            )  # 只考虑 non-cached 部分的 seq 作为 input_ids
            positions.extend(
                list(range(seq.num_cached_tokens, seqlen))
            )  # non-cache 部分的 seq pos_ids
            seqlen_q = seqlen - seq.num_cached_tokens  # non-cache 部分的 seq 也即  sl_q
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            """"
                cu_seqlens_q = [0, slq, 2*slq, ..]，即 batch 中 seqlen_q 以 checksum 形式的 平铺
            """
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
            """
                slot_mapping, 每个 seq 需要存储 kvcache 的 tokens 区间上 每个 token 的 start/end_block_id; 并在 batch 上平铺
            """
        print(
            f"[DEBUG prepare_prefill] Total input_ids: {len(input_ids)}, positions: {len(positions)}"
        )
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            print(f"[DEBUG prepare_prefill] Creating block tables")
            block_tables = self.prepare_block_tables(seqs)

        print(f"[DEBUG prepare_prefill] Creating tensors...")
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        print(f"[DEBUG prepare_prefill] Setting context...")
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        print(f"[DEBUG prepare_prefill] Completed")
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            # decode stage, 每次只加入 last-token 作为 input_ids
            positions.append(len(seq) - 1)
            # 其 next gen token 在 seq 上的绝对位置
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            """
                当前 input (亦即 seq.last_token) 在 kvcache 槽位:
                    blocks * block_size + last_block_num_tokens -1  
            """
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
            """
                Prefill 吐出首字
            """
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            """
                TODO: 
            """
            return self.model.compute_logits(graph_vars["outputs"][:bs])
            """
                decode 每计算一步，就转换一个 词表 token
            """

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        print(
            f"[DEBUG run] Starting run with {len(seqs)} sequences, is_prefill={is_prefill}"
        )
        print(f"[DEBUG run] Calling prepare_prefill/prepare_decode")
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        print(
            f"[DEBUG run] Input IDs shape: {input_ids.shape}, Positions shape: {positions.shape}"
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        print(f"[DEBUG run] Calling run_model")
        logits = self.run_model(input_ids, positions, is_prefill)
        print(f"[DEBUG run] Logits shape: {logits.shape}")
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )  # logits 采样 作为最终输出 tokens
        print(f"[DEBUG run] Token IDs: {token_ids}")
        reset_context()
        print(f"[DEBUG run] Run completed")
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
