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

    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    print("Tokenizer loaded successfully")

    print("Step 2: Initializing LLM engine...")
    start_time = time.time()
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"LLM engine initialized in {time.time() - start_time:.2f}s")

    print("Step 3: Testing generation...")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=10)
    prompts = ["Hello, how are you?"]

    print("Step 4: Starting generation...")
    outputs = llm.generate(prompts, sampling_params)

    print("Step 5: Results:")
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print("All steps completed successfully!")


if __name__ == "__main__":
    main()
