# eeLLaMA
--- 
Implementation of Meta AI's LLaMA model with modifications for early-exiting output networks. The ultimate goal is to deploy a head-model on some edge device for quick inference. This particular implementation will be optimized to run on MPS. In future iterations where tail models are designed to run in the cloud should be modified for CUDA support.