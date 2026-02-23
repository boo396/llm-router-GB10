#!/usr/bin/env bash
set -euo pipefail

# Re-image bootstrap for llm-router-GB10 (single GB10 flow)
# Usage:
#   bash scripts/reimage_bootstrap.sh
#
# After script completes:
#   1) Edit .env and demo/.env with real API keys if needed
#   2) docker compose up -d --build

REPO_URL="https://github.com/boo396/llm-router-GB10.git"
BRANCH="experimental"
REPO_DIR="llm-router-GB10"

echo "[1/8] Installing base packages"
sudo apt update
sudo apt install -y git curl ca-certificates gnupg lsb-release

echo "[2/8] Installing Docker + Compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  sudo apt install -y docker.io docker-compose-v2
fi
sudo systemctl enable --now docker

echo "[3/8] Installing NVIDIA Container Toolkit"
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  sudo apt update
  sudo apt install -y nvidia-container-toolkit
fi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "[4/8] Cloning repository"
if [[ ! -d "${REPO_DIR}" ]]; then
  git clone "${REPO_URL}"
fi

cd "${REPO_DIR}"
git fetch --all
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"

echo "[5/8] Building vLLM multimodal image"
sudo docker build -t vllm-openai-scipy -f docker/Dockerfile.vllm-scipy .

echo "[6/8] Creating env files"
cp -f demo/env_template.txt .env
cp -f demo/env_template.txt demo/.env

HOST_IP="$(hostname -I | awk '{print $1}')"
sed -i "s|^PHI4_MULTIMODAL_ENDPOINT=.*|PHI4_MULTIMODAL_ENDPOINT=http://${HOST_IP}:8012/v1|" .env demo/.env
sed -i "s|^PHI4_REASONING_ENDPOINT=.*|PHI4_REASONING_ENDPOINT=http://${HOST_IP}:8013/v1|" .env demo/.env
sed -i "s|^LOCAL_OPENAI_API_KEY=.*|LOCAL_OPENAI_API_KEY=local|" .env demo/.env

echo "[7/8] Starting Phi-4 reasoning container"
sudo docker rm -f phi4-reasoning >/dev/null 2>&1 || true
sudo docker run -d --name phi4-reasoning --gpus all \
  -p 8013:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --ipc=host \
  -e VLLM_USE_TRITON=0 -e VLLM_USE_FLASHINFER=0 \
  vllm/vllm-openai:latest \
  microsoft/phi-4 --dtype auto --trust-remote-code --max-model-len 16384 --enforce-eager

echo "[8/8] Creating Phi-4 multimodal container (then stopping it for single-GB10 memory)"
sudo docker rm -f phi4-multimodal >/dev/null 2>&1 || true
sudo docker run -d --name phi4-multimodal --gpus all \
  -p 8012:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --ipc=host \
  -e VLLM_USE_TRITON=0 -e VLLM_USE_FLASHINFER=0 \
  vllm-openai-scipy \
  microsoft/Phi-4-multimodal-instruct \
  --dtype auto --trust-remote-code --max-model-len 65536 \
  --limit-mm-per-prompt '{"audio":3,"image":3}' \
  --gpu-memory-utilization 0.85 --enforce-eager

# Give multimodal a short warm-up window, then stop to free GPU memory.
sleep 10
sudo docker stop phi4-multimodal >/dev/null 2>&1 || true

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  1) Edit .env and demo/.env for OPENAI_API_KEY / NVIDIA_API_KEY if you use those providers"
echo "  2) Start app stack: sudo docker compose up -d --build"
echo "  3) Open: http://${HOST_IP}:7860"
