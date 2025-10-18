import os
import argparse
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    args = parser.parse_args()

    path = os.path.expanduser(args.model_path)

    print("[DEBUG] Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    print("[DEBUG] Tokenizer loaded successfully")

    print("[DEBUG] Step 2: Initializing LLM engine...")
    start_time = time.time()
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"[DEBUG] LLM engine initialized in {time.time() - start_time:.2f}s")

    print("[DEBUG] Step 3: Setting up sampling parameters...")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    print("[DEBUG] Step 4: Preparing prompts...")
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    print("[DEBUG] Step 5: Starting generation...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"[DEBUG] Generation completed in {time.time() - start_time:.2f}s")

    print("[DEBUG] Step 6: Printing results...")
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print("[DEBUG] All steps completed successfully!")


if __name__ == "__main__":
    main()
