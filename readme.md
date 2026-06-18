wget https://raw.githubusercontent.com/tarabukinivan/albedo/refs/heads/main/pod_install.sh

chmod +x /root/albedo_run/pod_install.sh && bash -n /root/albedo_run/pod_install.sh && echo "syntax OK"

Когда скачаешь нового короля — всегда сначала inspect_champion.py, он скажет какой train запускать. Для большинства королей будет TIED или EXOTIC (как у arboshelper) → train_champion_tied.py. Если когда-то увидишь UNTIED — train_champion_untied.py

# Pod Setup — Albedo SN97 (RunPod A100/H100)

> Единый гайд установки и базовых операций на свежий RunPod образ
> **`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`**.
>
> Покрывает: установку зависимостей, Hippius login, скачивание моделей,
> upload модели на репозиторий. Используется как стартовая точка перед
> `finetune2/GUIDE.md` (свой SFT с untie) или `finetune3/GUIDE.md`
> (clone-with-improvement).

---

## 0. Заявка на pod

| Параметр | Значение |
|---|---|
| **Образ** | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| **GPU** | 1× **H100 80GB** (preferred) или 1× A100 80GB |
| **Disk** | **≥ 80 GB** (модели + checkpoints) |
| **RAM** | ≥ 32 GB (для CPU-side weight loading) |
| **Регион** | любой ближе к Hippius S3 (US-east обычно ОК) |

Стоимость:
- H100 80GB: ~$2.5–3.5/час
- A100 80GB: ~$1.5–2.5/час

Минимальное время аренды: 1 час (для finetune3) или 1.5 часа (для finetune2 + загрузка).

В образе уже стоят:
- Python 3.12
- torch 2.8.0 + CUDA 12.8
- nvidia driver, nvidia-smi

Нужно доустановить только Python пакеты — около ~5 минут.

---

## 1. Установка зависимостей

### 1.1 Автоматически

```bash
# Залить bundle с оркестратора (см. §3 о переносе)
cd /root
bash /root/albedo_run/pod_install.sh
```

Скрипт ставит всё одной командой и проверяет версии в конце.

### 1.2 Вручную (если хочешь видеть что ставится)

```bash
# 0. apt — tmux обязательно (без него SIGHUP убьёт тренировку)
apt-get update -qq && apt-get install -y -qq tmux rsync htop curl jq

# 1. Python core
pip install --quiet --upgrade pip
pip install --quiet \
    "transformers==5.12.1" \
    "trl==1.6.0" \
    "datasets>=2.20.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.5" \
    "pyarrow>=16.0.0" \
    "huggingface-hub>=0.25.0"

# 2. DeepSpeed (для multi-GPU; single-GPU не обязательно)
pip install --quiet "deepspeed>=0.15.0"

# 3. Hippius — для скачивания/upload моделей
pip install --quiet "hippius-hub>=0.5.0"

# 4. Bittensor — для register + commit on-chain
pip install --quiet \
    "bittensor>=10.0.0,<11" \
    "bittensor-wallet>=4.0.0,<5"

# 5. Utils
pip install --quiet "loguru" "pydantic>=2.8.0" "pydantic-settings>=2.4.0" "httpx>=0.27.0"
```

### 1.3 Опционально: flash-attention (~20 мин сборка)

```bash
# Это сборка из исходников, требует C++ toolchain. На некоторых образах не работает.
pip install --quiet flash-attn --no-build-isolation 2>&1 | tail -5
```

**Если флэш не ставится** — это ОК, используй `--no-flash-attn` (sdpa fallback) во всех train-скриптах. Разница в скорости на A100 ~20-30%, не критично.

### 1.4 Проверка

```bash
python3 -c "
import torch, transformers, trl, datasets, safetensors, hippius_hub
import bittensor as bt
print('torch:       ', torch.__version__, 'cuda:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('trl:         ', trl.__version__)
print('hippius_hub: ', hippius_hub.__version__ if hasattr(hippius_hub,'__version__') else 'OK')
print('bittensor:   ', bt.__version__)
"
```

Ожидаемый вывод:
```
torch:        2.8.0+cu128 cuda: True
transformers: 5.12.1
trl:          1.6.0
hippius_hub:  0.5.1
bittensor:    10.4.1
```

### 1.5 Albedo CLI

Для официальных команд upload/commit (включая v6 reveal):

```bash
git clone --depth 1 https://github.com/unarbos/albedo.git /root/albedo_repo
cd /root/albedo_repo
pip install -e .
albedo --help   # должна быть таблица команд: upload, commit, register, publish, …
```

---

## 2. Bittensor wallet — перенос на pod

Кошелек хранится на оркестраторе в `~/.bittensor/wallets/`. На pod'е нужны те же ключи (для register + commit):

```bash
# С оркестратора:
scp -r ~/.bittensor user@pod-ip:~/

# На pod'е проверить:
ls ~/.bittensor/wallets/runp/hotkeys/   # → должны быть aff1, aff2, ...
btcli wallet overview --wallet.name runp 2>&1 | head -10
```

⚠️ Никогда не передавай coldkey password по чату. Pod-провайдер может перехватить.

---

## 3. Перенос workspace (finetune2/3) на pod

```bash
# С оркестратора:
ssh user@pod-ip 'mkdir -p /root/albedo_run'
rsync -avz --exclude='ckpt' --exclude='*.safetensors' --exclude='*.log' \
  /root/albedo_run/ user@pod-ip:/root/albedo_run/

# На pod'е проверить:
ls /root/albedo_run/finetune2/data/sft/  # train_v3.jsonl должен быть
ls /root/albedo_run/finetune3/scripts/   # inspect_champion.py и т.д.
```

---

## 4. Hippius — настройка

### 4.1 Получение токенов

Зайди на `https://console.hippius.com` и:
1. Создай project (namespace) если его нет. Наш — **`tarab`** (уже provisioned, Free план, 100 GB).
2. В Settings → API Tokens → Create new → копируешь **consoleToken**.

### 4.2 Login на pod'е

```bash
# 1. Сохраняем consoleToken
export HIPPIUS_CONSOLE_TOKEN=70a4...   # твой токен

# 2. Login (сохраняет в ~/.cache/hippius/hub/token)
hippius-hub login --hippius-token "$HIPPIUS_CONSOLE_TOKEN"

# 3. (только для PUSH моделей) Сминтить registry robot-token —
#    consoleToken НЕ работает для push, нужен отдельный robot-secret.
hippius-hub registry rotate-token

# 4. Проверка
hippius-hub registry me
# Должно показать:
#   Project:   tarab
#   Plan:      Free
#   Status:    active
#   Quota:     100.0 GB
#   Login:     robot$tarab+tarab-bot
```

**Важное про токены:**

| Что | Когда нужно |
|---|---|
| `consoleToken` (только pull) | Скачать чужие модели (king, fingerprint checks) |
| `consoleToken` + `rotate-token` (push) | Upload своих моделей в `tarab/*` |

Robot-token хранится в `~/.cache/hippius/hub/token` и используется автоматически
для всех последующих `hippius-hub` и `albedo upload` команд.

### 4.3 Environment variable (опционально, для скриптов)

```bash
# Если хочешь чтобы env var автоматически передавался
export HIPPIUS_HUB_TOKEN=<console_token>
# Альтернатива username+password (на pod лучше токен):
# export HIPPIUS_HUB_USERNAME=...
# export HIPPIUS_HUB_PASSWORD=...
```

---

## 5. Скачивание базовых моделей

Для finetune2 (свой SFT) нужны три модели:

```bash
# Один-раз, занимает ~10-15 минут при норм. скорости
bash /root/albedo_run/finetune2/scripts/setup_pod.sh
```

Скрипт скачивает:
- `Qwen/Qwen3-4B` → `/workspace/models/Qwen3-4B` (genesis reference для arch lock)
- `Qwen/Qwen3-4B-Instruct-2507` → `/workspace/models/Qwen3-4B-Instruct-2507`
- Создаёт `/workspace/models/Qwen3-4B-Instruct-2507-overridden/` (config'и поправлены под genesis RoPE)

Для finetune3 (clone-from-champion) — только король:

```bash
python3 /root/albedo_run/finetune3/scripts/download_champion.py \
  --repo <king-namespace>/<king-name> \
  --revision main \
  --out /workspace/models/champion
```

Узнать актуальное имя короля:

```bash
curl -s https://s3.hippius.com/albedo/data/dashboard.json | python3 -c "
import json,sys
d = json.load(sys.stdin)
king = d['reign']['members'][0]
print(f'{king[\"model_uri\"]}  (king v{king[\"king_version\"]})')"
```

---

## 6. Upload модели в Hippius

Когда чекпойнт готов и решардён (после `train_*.py` + `reshard.py`):

### 6.1 Через `hippius-hub` (низкоуровневый, прозрачнее)

```bash
# Чтобы избежать 404 на больших блобах — снижаем concurrency:
export HIPPIUS_UPLOAD_WORKERS=1
export HIPPIUS_MAX_CONCURRENT=4
export HIPPIUS_CHUNK_SIZE=33554432   # 32 MB chunks

# Retry-цикл: Hippius merge-семантика дозальёт пропущенные шарды
for t in 1 2 3 4 5 6; do
  hippius-hub upload tarab/albedo-qwen3-4b-v3 \
    /root/albedo_run/finetune2/ckpt/v3/final_sharded \
    --revision main && break
  echo "Retry $t/6..."
  sleep 5
done

# Получить digest для последующего commit'a
hippius-hub revisions tarab/albedo-qwen3-4b-v3 --json | \
  grep -oE 'sha256:[a-f0-9]{64}' | head -1
# → sha256:abc123...
```

### 6.2 Через `albedo upload` (официальный CLI, рекомендуется)

```bash
cd /root/albedo_repo
albedo upload \
  --path /root/albedo_run/finetune2/ckpt/v3/final_sharded \
  --namespace tarab \
  --name v3
# Вернёт: repo=tarab/albedo-qwen3-4b-v3, digest=sha256:abc123...
```

Преимущество `albedo upload`:
- Проверяет repo_pattern (`^[^/]+/albedo-qwen3-4b-.+$`) перед upload'ом
- Сразу печатает digest и reveal-строку для commit'a
- Использует ту же retry-логику что и `hippius-hub` напрямую

### 6.3 Проверка что upload удался

```bash
# Через CLI:
albedo check-hippius --repo tarab/albedo-qwen3-4b-v3 --digest sha256:abc123...
# → VALID если файлы прошли file_manifest + architecture check

# Или вручную — увидеть структуру файлов:
python3 -c "
import hippius_hub as h
api = h.HippiusApi()
files = api.list_repo_files(repo_id='tarab/albedo-qwen3-4b-v3', revision='main')
for f in files: print(' ', f.path if hasattr(f,'path') else f)
"
```

---

## 7. Download чужой модели (для eval / clone start)

### 7.1 Через `hippius_hub` Python API

```python
import hippius_hub as h
api = h.HippiusApi()
api.hf_hub_download(
    repo_id="arboshelper/albedo-qwen3-4b-2-5-final",
    filename="model.safetensors",
    revision="main",
    local_dir="/workspace/models/champion",
)
```

### 7.2 Через `download_champion.py` (рекомендуется)

См. §5 — обработает все нужные файлы автоматически.

### 7.3 Через `albedo` CLI (только manifest+config check)

```bash
albedo check-hippius --repo arboshelper/albedo-qwen3-4b-2-5-final \
  --digest sha256:8b823f...
# Скачает только config.json и проверит arch lock; веса не качает.
```

---

## 8. Регистрация и commit on-chain

⚠️ Это **тратит TAO** (~τ1-3 за регистрацию). Делать только когда чекпойнт готов и проверен локально.

```bash
cd /root/albedo_repo

# 1. Проверить баланс
python3 -c "
import bittensor as bt
st = bt.Subtensor(network='finney')
w = bt.Wallet(name='runp', hotkey='aff2')
print(f'recycle: {st.recycle(netuid=97)}')
print(f'balance: {st.get_balance(w.coldkeypub.ss58_address)}')
"

# 2. Регистрация нового хоткея (TAO burn, необратимо)
albedo register --coldkey runp --hotkey aff2 --netuid 97

# 3. Commit v6 reveal
albedo commit \
  --repo tarab/albedo-qwen3-4b-v3 \
  --digest sha256:abc123... \
  --coldkey runp --hotkey aff2

# 4. Проверить что commit в цепочке
albedo check-commit --hotkey 5FJB...   # ss58 нашего hotkey
```

⚠️ Перед каждым commit'ом — НОВЫЙ hotkey (правило сабнета: 1 commit = 1 eval = 1 hotkey). Например aff1 → v2, aff2 → v3, aff3 → champ1, ...

---

## 9. Диагностика типичных проблем

| Симптом | Причина | Решение |
|---|---|---|
| `Intel oneMKL FATAL ERROR libtorch_cpu.so` | Транзиент контейнера | Повторить команду |
| `transformers KeyError 'rope_parameters'` | Не активирован шим в скрипте | Использовать наши train_*.py — там шим встроен |
| `CUDA OOM` | seq=8192 + flash-attn off → eager OOM | `--no-flash-attn` использует sdpa, не eager |
| `hippius-hub 401 unauthorized` | Прошёл только `login`, нужен `rotate-token` | Запустить `hippius-hub registry rotate-token` |
| `hippius-hub 404 на 8 GB блобе` | Реестр флаки на больших блобах | Решардить в 5×~2 GB (`reshard.py`) и повторить upload |
| `albedo commit: insufficient balance` | Балансом coldkey ниже recycle | Пополнить coldkey или ждать снижения цены |
| `albedo register: hotkey already registered` | Хоткей был ранее на 97 → не подходит для нового commit | Использовать другой свободный хоткей (aff3, ...) |
| `ImportError: flash_attn` при загрузке model | Не указан `--no-flash-attn` | Добавить флаг |
| Тренировка умерла на step N | SIGHUP при закрытии SSH | Всегда `tmux new-session -d ...` |

---

## 10. Чек-лист после сетапа пода

- [ ] `nvidia-smi` показывает 1× H100/A100 80GB, ≥ 80GB свободно
- [ ] `python3 -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `pip show transformers` → 5.12.1
- [ ] `pip show trl` → 1.6.0
- [ ] `hippius-hub registry me` → Project: tarab, Status: active
- [ ] `albedo --help` показывает таблицу команд
- [ ] `ls ~/.bittensor/wallets/runp/hotkeys/` есть aff2, aff3 и т.д.
- [ ] `tmux` установлен (`which tmux`)
- [ ] `/workspace/models/` создан, свободно ≥ 30 GB

---

## 11. Финальная картина рабочего пода

```
/root/
├── albedo_run/                       ← workspace (rsync с оркестратора)
│   ├── POD_SETUP.md                  ← этот файл
│   ├── pod_install.sh                ← автоматический installer
│   ├── myalbedo.md                   ← главная боевая дока
│   ├── finetune1/                    v2 chal-only SFT (для reference)
│   ├── finetune2/                    v3: untie + completion-only SFT
│   └── finetune3/                    clone-with-improvement
├── albedo_repo/                      ← официальный github.com/unarbos/albedo
│   └── (предоставляет `albedo` CLI)
└── .bittensor/wallets/runp/          ← кошельки (rsync с оркестратора)

/workspace/models/                    ← базовые модели + чемпионы (8-20 GB каждая)
├── Qwen3-4B/                         genesis (для arch lock reference)
├── Qwen3-4B-Instruct-2507/           база для finetune2
├── Qwen3-4B-Instruct-2507-overridden/  готова к SFT
└── champion/                         текущий король (для finetune3)
```

---

## 12. Quick-start templates

### A. Сетап пода с нуля → готов к тренировке (15-20 мин)

```bash
# 1. На оркестраторе (1×):
rsync -avz --exclude='ckpt' --exclude='*.safetensors' \
  /root/albedo_run/ user@pod:/root/albedo_run/
scp -r ~/.bittensor user@pod:~/

# 2. На pod'e
ssh user@pod
bash /root/albedo_run/pod_install.sh           # ~5 мин
hippius-hub login --hippius-token <token>
hippius-hub registry rotate-token              # для upload
bash /root/albedo_run/finetune2/scripts/setup_pod.sh    # ~10 мин: скачка моделей
```

### B. Запуск finetune2 (~30-40 мин)

```bash
cd /root/albedo_run/finetune2
tmux new-session -d -s train2 -c .
tmux send-keys -t train2 'python3 scripts/train_v3.py \
  --data data/sft/train_v3.jsonl \
  --output ckpt/v3 \
  --base /workspace/models/Qwen3-4B-Instruct-2507-overridden \
  --epochs 1 --lr 1e-5 --batch-size 2 --grad-accum 8 \
  --no-flash-attn \
  2>&1 | tee /tmp/train_v3.log; echo "EXIT=${PIPESTATUS[0]}"' Enter
tmux attach -t train2     # (Ctrl+b d = detach)
```

### C. Запуск finetune3 (~10-20 мин total)

```bash
cd /root/albedo_run/finetune3

# 1. Узнать короля
KING=$(curl -s https://s3.hippius.com/albedo/data/dashboard.json | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['reign']['members'][0]['model_uri'].split('@')[0])")
echo "King: $KING"

# 2. Скачать
python3 scripts/download_champion.py --repo "$KING" --out /workspace/models/champion

# 3. Inspect — определить tied / exotic / untied
python3 scripts/inspect_champion.py --champion-dir /workspace/models/champion

# 4. По вердикту запустить НУЖНЫЙ train (см. GUIDE.md):
tmux new-session -d -s champ -c .
tmux send-keys -t champ 'python3 scripts/train_champion_tied.py \
  --champion-dir /workspace/models/champion \
  --data data/sft/train_v3.jsonl \
  --output ckpt/champ \
  --lr 5e-7 --max-steps 60 --save-steps 10 \
  --no-flash-attn \
  2>&1 | tee /tmp/champ_train.log; echo "EXIT=${PIPESTATUS[0]}"' Enter
```

### D. После тренировки → Hippius upload (~5-10 мин)

```bash
# 1. Решард в 5×2GB (обязательно для Hippius)
python3 finetune2/scripts/reshard.py \
  --src finetune2/ckpt/v3/final \
  --dst finetune2/ckpt/v3/final_sharded \
  --max-shard 2GB

# 2. Локальная валидация
python3 finetune2/scripts/validate_local.py \
  --model-dir finetune2/ckpt/v3/final_sharded \
  --repo tarab/albedo-qwen3-4b-v3 \
  --king-dir /workspace/models/Qwen3-4B \
  --fingerprint-against /workspace/models/Qwen3-4B \
    /workspace/models/Qwen3-4B-Instruct-2507 \
    /workspace/models/Qwen3-4B-Instruct-2507-overridden

# 3. Upload в Hippius
cd /root/albedo_repo
albedo upload \
  --path /root/albedo_run/finetune2/ckpt/v3/final_sharded \
  --namespace tarab --name v3
# → запиши digest
```

### E. Commit on-chain (5 мин, тратит TAO)

```bash
albedo register --coldkey runp --hotkey aff2 --netuid 97
albedo commit \
  --repo tarab/albedo-qwen3-4b-v3 \
  --digest sha256:<DIGEST_FROM_UPLOAD> \
  --coldkey runp --hotkey aff2
albedo check-commit --hotkey <SS58_AFF2>   # проверить
```

---

## 13. Завершение pod'а

После commit'а pod больше не нужен — модель уже в Hippius, валидатор её скачает сам.

```bash
# 1. Забрать с пода важные артефакты (на случай v4):
#    Только конфиги + логи, веса уже в Hippius
rsync -avz --exclude='*.safetensors' --exclude='checkpoint-*' \
  user@pod:/root/albedo_run/finetune2/ckpt/ \
  /root/albedo_run/finetune2_backup/

# 2. Pod выключить через RunPod UI — иначе будет тикать счёт.
```
