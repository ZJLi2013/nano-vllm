import os
import argparse
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
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
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
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
