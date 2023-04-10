import torch 
from eellama import TailTransformer, HeadTransformer, ModelArgs, Tokenizer
import math

CHECKPOINT = "/Users/chase/workspace/ai_project/llama_experiments/checkpoints/7B/consolidated.00.pth"
TOKENIZER = "/Users/chase/workspace/ai_project/llama_experiments/checkpoints/tokenizer.model"

# Only works for single .pth file currently
def split_state_dict(total_layers: int, split: int):
	# Compute num layers
	num_layers = total_layers // split

	# Load original state dict
	checkpoint = torch.load(CHECKPOINT)

	# Create new state dict
	head_state_dict = {}
	tail_state_dict = {}

	# Copy embeddings weights to the head and output/final norm weights to the tail
	head_state_dict["tok_embeddings.weight"] = checkpoint["tok_embeddings.weight"]
	tail_state_dict["norm.weight"] = checkpoint["norm.weight"]
	tail_state_dict["output.weight"] = checkpoint["output.weight"]
		
	for key in checkpoint.keys():
		if "layers" not in key: continue

		layer_id = int(key.split(".")[1])
		if layer_id < num_layers:
			head_state_dict[key] = checkpoint[key]
		else:
			tail_state_dict[key] = checkpoint[key]	
	
	return head_state_dict, tail_state_dict

def main():
	# Create new checkpoint
	# Creats a head with 32 / 8 = 4 layers and a tail with 32 - 4 = 28 layers
	head_dict, tail_dict = split_state_dict(total_layers=32, split=8)

	# Load tokenizer
	tokenizer = Tokenizer(model_path=TOKENIZER)

	# Model Configurations
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

	tail_args: ModelArgs = ModelArgs(
		max_seq_len=2048,
		max_batch_size=32,
		n_layers = 28,
		n_heads = 32,
		dim=4096,
		multiple_of=256, 
		norm_eps=1e-06,
		vocab_size=tokenizer.n_words,
	)

	# float16
	torch.set_default_tensor_type(torch.HalfTensor)

	head_model = HeadTransformer(head_args).to("cpu")
	tail_model = TailTransformer(tail_args).to("cpu")
	head_model.load_state_dict(head_dict, strict=False)
	tail_model.load_state_dict(tail_dict, strict=False)

	# Save model state dicts
	torch.save(head_model.state_dict(), "../model_out/head_checkpoint.pth")
	torch.save(tail_model.state_dict(), "../model_out/tail_checkpoint.pth")

	# Save head model to torchscript
	head_model_scripted = torch.jit.script(head_model)
	head_model_scripted.save("../model_out/head_model_scripted.pt")
	
	"""
	model.to("mps")
	model.eval()

	# Test one token generation
	prompt = "Hello, my name is"
	encoded = tokenizer.encode(prompt, bos=True, eos=False)
	tokens = torch.full((1, 50), tokenizer.pad_id).to("mps").long()
	tokens[0, : len(encoded)] = torch.tensor(encoded).to("mps").long()

	min_prompt_size = len(encoded)
	start_pos = min_prompt_size
	output = model.forward(tokens[:, 0:start_pos], 0)

	# Output model to torchscript
	torch.save(model.state_dict(), "../model_out/head_checkpoint.pth")
	# model_scripted = torch.jit.script(model)
	# model_scripted.save("../model_out/model_scripted.pt")
	"""



if __name__ == "__main__":
	main()