#!/bin/bash
# pod_install.sh — установка всех зависимостей на свежий RunPod A100/H100.
#
# Целевой образ: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   (Python 3.12, torch 2.8.0+cu128, CUDA 12.8 уже стоят)
#
# Что ставится:
#   - transformers 5.12.1 (HF, нужно для Qwen3 v5-формат RoPE)
#   - trl 1.6.0 (SFT trainer + completion_only_loss)
#   - datasets, accelerate, safetensors, pyarrow
#   - hippius-hub (для скачивания/upload моделей)
#   - bittensor 10.x + bittensor-wallet (для register/commit)
#   - albedo CLI (из официального репо)
#
# НЕ ставится автоматически:
#   - flash-attn (нужна сборка из исходников, 15-30 мин). Опционально.
#     Использовать --no-flash-attn в train_*.py если без него.
#
# Запуск:
#   bash /root/albedo_run/pod_install.sh
# или ALBEDO_NO_FLASH=1 для пропуска даже попытки flash-attn:
#   ALBEDO_NO_FLASH=1 bash /root/albedo_run/pod_install.sh

set -euo pipefail
cd /root

# === 1. Sanity: проверка образа ===
echo "=== 1. Image sanity check ==="
python3 --version
echo -n "  torch: "
python3 -c "import torch; print(torch.__version__, '+ CUDA' if torch.cuda.is_available() else '+ NO CUDA')"
echo -n "  GPU: "
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "no GPU"
echo

# === 2. Базовые системные пакеты (если что-то не установлено) ===
echo "=== 2. apt deps (tmux, rsync, htop) ==="
apt-get update -qq 2>&1 | tail -3
apt-get install -y -qq tmux rsync htop curl jq 2>&1 | tail -3
echo

# === 3. Python deps — pinned versions ===
echo "=== 3. Python packages (pinned) ==="
pip install --quiet --upgrade pip

# Group 1: ML core (transformers v5 для Qwen3 RoPE формата)
pip install --quiet \
    "transformers==5.12.1" \
    "trl==1.6.0" \
    "datasets>=2.20.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.5" \
    "pyarrow>=16.0.0" \
    "huggingface-hub>=0.25.0"

# Group 2: DeepSpeed (опционально, для multi-GPU)
pip install --quiet "deepspeed>=0.15.0" 2>&1 | tail -3 || echo "  [warn] deepspeed install failed (ok if single-GPU)"

# Group 3: Hippius (для download/upload моделей)
pip install --quiet "hippius-hub>=0.5.0"

# Group 4: Bittensor (для register/commit)
pip install --quiet \
    "bittensor>=10.0.0,<11" \
    "bittensor-wallet>=4.0.0,<5"

# Group 5: Utils
pip install --quiet \
    "loguru" "pydantic>=2.8.0" "pydantic-settings>=2.4.0" \
    "httpx>=0.27.0" "openai>=1.40.0"

echo

# === 4. Verify imports ===
echo "=== 4. Verify imports ==="
python3 - <<'PYEOF'
import torch, transformers, trl, datasets, accelerate, safetensors, hippius_hub
import bittensor as bt
print(f"  torch:        {torch.__version__}  cuda: {torch.cuda.is_available()}")
print(f"  transformers: {transformers.__version__}")
print(f"  trl:          {trl.__version__}")
print(f"  datasets:     {datasets.__version__}")
print(f"  accelerate:   {accelerate.__version__}")
print(f"  safetensors:  {safetensors.__version__}")
print(f"  hippius_hub:  {hippius_hub.__version__ if hasattr(hippius_hub,'__version__') else 'OK'}")
print(f"  bittensor:    {bt.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU{i}: {p.name}  {p.total_memory/1024**3:.1f} GB")
PYEOF
echo

# === 5. Clone official albedo repo (для CLI команд upload/commit) ===
echo "=== 5. Clone official albedo repo ==="
if [ ! -d /root/albedo_repo ]; then
    git clone --depth 1 https://github.com/unarbos/albedo.git /root/albedo_repo
fi
cd /root/albedo_repo
# Install in current venv (provides `albedo` console-script)
pip install --quiet -e . 2>&1 | tail -3
which albedo || true
echo

# === 6. Flash-attention (опционально) ===
if [ "${ALBEDO_NO_FLASH:-0}" != "1" ]; then
    echo "=== 6. flash-attn (опционально, build ~20 мин) ==="
    if python3 -c "import flash_attn" 2>/dev/null; then
        echo "  flash_attn уже стоит — skip"
    else
        echo "  Попытка установки. Если долго — Ctrl+C и используй --no-flash-attn в train скриптах."
        pip install --quiet flash-attn --no-build-isolation 2>&1 | tail -3 || \
            echo "  [warn] flash-attn install failed — это ОК, используй --no-flash-attn"
    fi
else
    echo "=== 6. flash-attn (SKIPPED via ALBEDO_NO_FLASH=1) ==="
fi
echo

# === 7. Workspace dirs ===
echo "=== 7. Workspace dirs ==="
mkdir -p /workspace/models /root/albedo_run/{ckpt,logs}
echo "  /workspace/models  (для базовых моделей и чемпионов)"
echo "  /root/albedo_run/ckpt /logs"
echo

echo "=================================================================="
echo "✅ Install complete."
echo
echo "Next steps:"
echo "  1. Скачай базовые модели:"
echo "       bash /root/albedo_run/finetune2/scripts/setup_pod.sh"
echo "     (создаст /workspace/models/Qwen3-4B + Qwen3-4B-Instruct-2507 + overridden)"
echo
echo "  2. Hippius login (нужен для upload):"
echo "       hippius-hub login --hippius-token <CONSOLE_TOKEN>"
echo "       hippius-hub registry rotate-token   # робот-токен для push"
echo "       hippius-hub registry me             # проверка"
echo
echo "  3. Tip: если хочешь скачать существующего короля для finetune3 / eval —"
echo "     для PULL'а consoleToken достаточно (без registry rotate)."
echo "=================================================================="
