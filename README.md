# SpliTEE - Improving LLM Inference on Trusted Hardware with Differentially Private GPU Outsourcing
This repository contains the main code for our privacy-preserving split inference framework for Large Language Models (LLMs) using a Trusted Execution Environment (TEE) and GPU offloading.


## Method
<p align="justify">
Running an entire LLM inside a CPU-based TEE can introduce significant computational overhead. Our approach therefore divides LLM inference between an Intel TDX virtual machine and an external GPU. The TDX environment retains privacy-sensitive operations such as tokenization, the embedding layer, and normalization operations, while computationally expensive linear operations are outsourced to the GPU. These outsourced operations include the Q, K, V, and O projections in the attention block, the Gate and Up projections in the MLP block, and the LM head. To protect intermediate representations exposed to the GPU, the TDX environment masks an operator input (x) using additive Gaussian noise (η): x̃ = x + η. The masked input is sent to the GPU, which computes: W x̃ = W(x + η). The corresponding correction (Wη) is then used inside the trusted environment to recover the intended result: Wx = W x̃ - Wη. The masking noise is calibrated using operator-specific sensitivity bounds and a configurable privacy parameter (ϵ)
</p>

<img width="901" height="321" alt="image" src="https://github.com/user-attachments/assets/4ddc528a-469b-43a2-9166-e2ca419eec35" />


## Software Environment

The main experiments were conducted using the following software versions:

- Python 3.10.20
- NumPy 2.2.6
- pandas 2.3.3
- Matplotlib 3.10.9
- PyTorch 2.7.1+cu128
- Transformers 5.12.1
- OpenAI 2.50.0
- sentence-transformers 5.6.1
- tqdm 4.68.3
- CUDA 12.8
