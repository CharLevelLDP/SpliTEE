# SpliTEE - Improving LLM Inference on Trusted Hardware with Differentially Private GPU Outsourcing
This repository contains the main code for our privacy-preserving split inference framework for Large Language Models (LLMs) using a Trusted Execution Environment (TEE) and GPU offloading.


## Method

Running an entire LLM inside a CPU-based TEE can introduce significant computational overhead. Our approach therefore divides LLM inference between an Intel TDX virtual machine and an external GPU.

The TDX environment retains privacy-sensitive operations such as tokenization, the embedding layer, and normalization operations, while computationally expensive linear operations are outsourced to the GPU. 
These outsourced operations include the Q, K, V, and O projections in the attention block, the Gate and Up projections in the MLP block, and the LM head.

To protect intermediate representations exposed to the GPU, the TDX environment masks an operator input (x) using additive Gaussian noise (η): x̃ = x + η

The masked input is sent to the GPU, which computes: W x̃ = W(x + η)

The corresponding correction (Wη) is then used inside the trusted environment to recover the intended result: Wx = W x̃ - Wη

The masking noise is calibrated using operator-specific sensitivity bounds and a configurable privacy parameter (ϵ)
