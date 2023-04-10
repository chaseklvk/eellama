from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from pathlib import Path

from eellama import ModelArgs, HeadTransformer, Tokenizer, LLaMA

def load(
    ckpt_path: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
	start_time = time.time()

	print("Loading")
	checkpoint = torch.load(ckpt_path, map_location="mps")
	tokenizer = Tokenizer(model_path=tokenizer_path)

	head_args: ModelArgs = ModelArgs(
		max_seq_len=2048,
		max_batch_size=32,
		n_layers = 4,
		n_heads = 32,
		dim=4096,
		multiple_of=256, 
		norm_eps=1e-06,
		vocab_size=tokenizer.n_words,
	)

	torch.set_default_tensor_type(torch.HalfTensor)
	
	model = HeadTransformer(head_args)
	model.load_state_dict(checkpoint, strict=False)

	generator = LLaMA(model, tokenizer)
	generator.model.to("mps")
	generator.model.eval()
	print(f"Loaded in {time.time() - start_time:.2f} seconds")
	return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    generator = load(
        ckpt_dir, tokenizer_path, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "Hello, I am",
    ]
    results = generator.generate(
        prompts, max_gen_len=50, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    main(
		ckpt_dir="/Users/chase/workspace/ai_project/llama_experiments/models/model_out/head_checkpoint.pth",
		tokenizer_path="/Users/chase/workspace/ai_project/llama_experiments/checkpoints/tokenizer.model"
	)