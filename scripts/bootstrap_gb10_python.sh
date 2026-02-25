#!/usr/bin/env bash
set -euo pipefail

# GB10-safe Python bootstrap for local (non-Docker) development.
# Override defaults if your host uses a different CUDA/PyTorch build.
#
# USE_UV mode quick reference:
#   USE_UV=0    -> pip-only (default), no uv fallback
#   USE_UV=auto -> fallback to uv only when pip times out
#   USE_UV=1    -> fallback to uv on timeout or resolver conflict

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install "setuptools<81"

CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-constraints/gb10-python312.txt}"
PIP_CONSTRAINT_ARGS=()
if [[ -f "$CONSTRAINTS_FILE" ]]; then
  echo "[bootstrap_gb10_python] Using constraints: ${CONSTRAINTS_FILE}"
  PIP_CONSTRAINT_ARGS=(-c "$CONSTRAINTS_FILE")
else
  echo "[bootstrap_gb10_python] Constraints file not found; continuing without constraints"
fi

PIP_RESOLVE_TIMEOUT_SEC="${PIP_RESOLVE_TIMEOUT_SEC:-900}"
USE_UV="${USE_UV:-0}"

TORCH_VERSION="${TORCH_VERSION:-2.9.1+cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_GB10_TORCH="${INSTALL_GB10_TORCH:-1}"

echo "[bootstrap_gb10_python] mode USE_UV=${USE_UV} timeout=${PIP_RESOLVE_TIMEOUT_SEC}s install_torch=${INSTALL_GB10_TORCH} constraints=${CONSTRAINTS_FILE}"

if [[ "$INSTALL_GB10_TORCH" == "1" ]]; then
  echo "[bootstrap_gb10_python] Installing PyTorch (${TORCH_VERSION}) from ${TORCH_INDEX_URL}"
  if ! pip install --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"; then
    echo "[bootstrap_gb10_python] GB10 torch install failed; falling back to default resolver (torch>=2.3)"
    pip install "torch>=2.3"
  fi
fi

run_pip_install() {
  if command -v timeout >/dev/null 2>&1; then
    if ! timeout --foreground "${PIP_RESOLVE_TIMEOUT_SEC}" \
      pip install "${PIP_CONSTRAINT_ARGS[@]}" --no-build-isolation -e .; then
      local status=$?
      if [[ "$status" -eq 124 ]]; then
        echo "[bootstrap_gb10_python] dependency resolution timed out after ${PIP_RESOLVE_TIMEOUT_SEC}s"
        return 124
      fi
      echo "[bootstrap_gb10_python] dependency install failed"
      return 1
    fi
  else
    if ! pip install "${PIP_CONSTRAINT_ARGS[@]}" --no-build-isolation -e .; then
      echo "[bootstrap_gb10_python] dependency install failed"
      return 1
    fi
  fi
  return 0
}

run_uv_fallback() {
  echo "[bootstrap_gb10_python] USE_UV=1; attempting uv fallback"
  if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap_gb10_python] uv not found; installing into current venv"
    pip install uv
  fi

  if ! uv sync --python 3.12; then
    echo "[bootstrap_gb10_python] uv sync failed"
    return 1
  fi

  return 0
}

if ! run_pip_install; then
  pip_status=$?
  if [[ "$USE_UV" == "1" ]] || [[ "$USE_UV" == "auto" && "$pip_status" -eq 124 ]]; then
    if ! run_uv_fallback; then
      echo "[bootstrap_gb10_python] Both pip and uv bootstrap paths failed"
      echo "[bootstrap_gb10_python] Recommended fallback: Docker flow for this repo"
      exit 1
    fi
  else
    if [[ "$pip_status" -eq 124 ]]; then
      echo "[bootstrap_gb10_python] Try tighter constraints, set USE_UV=1, or use Docker workflow"
      exit 2
    fi
    if [[ "$USE_UV" == "auto" ]]; then
      echo "[bootstrap_gb10_python] Resolver conflict detected before timeout; USE_UV=auto does not fallback on conflict"
      echo "[bootstrap_gb10_python] Re-run with USE_UV=1 to force uv fallback, or use Docker workflow"
      exit 1
    fi
    echo "[bootstrap_gb10_python] If this is a resolver conflict, set USE_UV=1 (or USE_UV=auto for timeout-only fallback) or use Docker workflow"
    exit 1
  fi
fi

echo
echo "Bootstrap complete."
echo "Run: source .venv/bin/activate && ./scripts/run_local.sh"
