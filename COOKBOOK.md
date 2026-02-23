# LLM Router GB10 Cookbook (From Scratch)

This cookbook recreates the local Phi-4 reasoning + Phi-4 multimodal routing setup on a single GB10. It includes the model switcher behavior and the UI warning about startup delays.

## 1) Prerequisites

- Ubuntu/Linux host
- NVIDIA driver installed and working (`nvidia-smi`)
- Docker + Docker Compose
- NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- Git

## 2) Clone the repo

```bash
git clone https://github.com/boo396/llm-router-GB10.git
cd llm-router-GB10
```

## 3) Configure Docker GPU runtime

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU in containers:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## 4) Build the vLLM multimodal image with SciPy

```bash
sudo docker build -t vllm-openai-scipy -f docker/Dockerfile.vllm-scipy .
```

## 5) Start the reasoning model (Phi-4)

> ⚠️ Reasoning and multimodal models cannot run concurrently on a single GB10 without OOM. The router auto-switches and will warn about delays.

```bash
sudo docker run -d --name phi4-reasoning --gpus all \
  -p 8013:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  -e VLLM_USE_TRITON=0 -e VLLM_USE_FLASHINFER=0 \
  vllm/vllm-openai:latest \
  microsoft/phi-4 --dtype auto --trust-remote-code --max-model-len 16384 --enforce-eager
```

## 6) Start the multimodal model (Phi-4 Multimodal)

> Start this only when you need multimodal, or let the router auto-switch.

```bash
sudo docker run -d --name phi4-multimodal --gpus all \
  -p 8012:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  -e VLLM_USE_TRITON=0 -e VLLM_USE_FLASHINFER=0 \
  vllm-openai-scipy \
  microsoft/Phi-4-multimodal-instruct \
  --dtype auto --trust-remote-code --max-model-len 65536 \
  --limit-mm-per-prompt '{"audio":3,"image":3}' \
  --gpu-memory-utilization 0.85 --enforce-eager
```

## 7) Configure environment variables

Create `.env` and `demo/.env` (do not commit):

```bash
cp demo/env_template.txt demo/.env
cp demo/env_template.txt .env
```

Edit both and set:

```
PHI4_MULTIMODAL_ENDPOINT=http://192.168.1.6:8012/v1
PHI4_REASONING_ENDPOINT=http://192.168.1.6:8013/v1
LOCAL_OPENAI_API_KEY=local
```

(Optional) Azure/NVIDIA keys for fallback.

## 8) Start router + demo UI

```bash
sudo docker compose up -d --build
```

Open the UI:

```
http://192.168.1.6:7860
```

## 9) Test routing

Text prompt (routes to Phi-4 reasoning):

```
Solve this: If A implies B and B implies C, what follows?
```

Image prompt (routes to Phi-4 multimodal):

```
Tell me about this picture.
```

## 10) Notes

- The router auto-switches containers and logs warnings about delay. This is expected on a single GB10.
- For seamless routing without delays, add another GB10 so both models run concurrently.
- Image uploads are resized/compressed to prevent token overflow.

## 11) Cleanup

```bash
sudo docker stop phi4-reasoning phi4-multimodal
sudo docker rm phi4-reasoning phi4-multimodal
sudo docker compose down
```
