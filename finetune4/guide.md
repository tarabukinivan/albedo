# Albedo finetune4 — Qwen3.6-35B-A3B MoE (DRAFT)

> **STATUS: PREMATURE (19.06 утром)** — chain.toml на main репо ещё не обновлён.
> Финальный repo_pattern + seed_digest + ALL_LOCK_KEYS пока неизвестны.
> Этот гайд — **подготовительный**. Полная активация после выкатки.

---

## 0. ⚠️ Что обязательно нужно проверить ДО запуска

1. **chain.toml** — `cd /root/albedo && git pull` затем
   ```bash
   grep -A20 "\[chain\]\|\[arch\]\|\[seed\]" /root/albedo/chain.toml
   ```
   Должно быть `seed_repo = "teutonic/qwen3.6-35b-a3b-genesis"` и **новые
   extra_lock_keys** (вероятно включают `num_experts`, `moe_intermediate_size`).

2. **architecture.py** — проверить как validator теперь сверяет lock keys для
   `text_config`-вложенных полей:
   ```bash
   cat /root/albedo/src/config_validation/checks/architecture.py | head -50
   ```

3. **rubric/sanity** — все ещё работают:
   ```bash
   cat /root/albedo/src/sanity_service/prompts.json
   # Должно: ["double(x)", "add(a,b)", "print(x[5])"]
   ```

4. **win_margin** — на dev `8d56680` поднят с 0.02 → 0.06:
   ```bash
   grep CHALLENGER_WIN_MARGIN /root/albedo/src/albedo_eval_service/judge_core.py
   # Должно: CHALLENGER_WIN_MARGIN = 0.06
   ```

---

## 1. Что в папке

```
finetune4/
├── GUIDE.md                              ⭐ этот файл
├── refs/
│   ├── qwen36_35b_genesis_config.json    config нового genesis (для anal)
│   ├── qwen36_35b_genesis_index.json     index 26 шардов / 1045 тензоров
│   ├── expected_arch_lock_v4.md          предсказание ALL_LOCK_KEYS до выкатки
│   └── judge_prompts_recovered.md        PROBE_SYSTEM (recovered, для anti-injection в SFT)
├── data/sft/
│   ├── train_v4.jsonl                    создаётся build_sft_v4.py
│   └── red_team.jsonl                    скопировать из finetune3 если нужен
├── configs/
│   └── ds_zero3.json                     ZeRO-3 с CPU offload (нужно для 35B)
└── scripts/
    ├── download_genesis_v4.py            ⭐ скачать новый genesis (72 GB) из Hippius
    ├── build_sft_v4.py                   ⭐ перетокенизация train_v3 под Qwen3.6 tokenizer
    ├── train_v4.py                       ⭐ SFT MoE 35B (DeepSpeed ZeRO-3)
    ├── reshard_v4.py                     решард в 35+ шардов по 2 GB (~70 GB total)
    ├── fingerprint_check.py              prod-identical clone-detection
    └── eval_local_v4.py                  ⭐ duel emulator через vLLM endpoints
```

---

## 2. Аппаратные требования

| Что | Min | Реалистично |
|---|---|---|
| **GPU** | 2× H100 80GB | **4× H100 80GB** (или 4× A100 80GB) |
| **Disk** | 200 GB | **300 GB** (genesis 72 + ckpt 72 + reshard 72 + кэш + dataset) |
| **RAM** | 64 GB | **128 GB** (для CPU offload в ZeRO-3) |
| **Network** | 1 Gbps | 10 Gbps (faster Hippius download) |
| **Pod cost** | $8-12/час (2×H100) | **$15-25/час (4×H100)** |

**ETA одного цикла**:
- Download genesis: 10-20 мин
- SFT 200 steps на 4×H100: ~2-4 часа
- Reshard: 10-15 мин
- Upload в Hippius: 15-30 мин (72 GB → 35 шардов)
- **Итого: 4-6 часов / итерация ≈ $80-150**

vs finetune2 (Qwen3-4B) — ~30 мин / итерация ≈ $5.

---

## 3. Что отличается от finetune2/finetune3 (Qwen3-4B)

| Аспект | Qwen3-4B (finetune2/3) | **Qwen3.6-35B-A3B (finetune4)** |
|---|---|---|
| Архитектура | Qwen3ForCausalLM | **Qwen3_5MoeForConditionalGeneration** (MoE + vision) |
| total params | 4.022 B | **35 B** (3 B active) |
| Шарды чекпойнта | 5 × ~2 GB | **35+ × ~2 GB** |
| GPU | 1× H100 80GB | **4× H100 80GB + ZeRO-3** |
| Tokenizer | vocab 151936 | **vocab 248320** (другой!) |
| Loss | completion_only_loss | то же (если TRL поддержит multimodal) |
| Vision | нет | **есть** — frozen для SWE (см. `--freeze-vision` в train_v4.py) |
| MoE | нет | 256 experts, top-k 8 |
| MTP | нет | 1 layer (multi-token prediction) |
| Hybrid attention | нет | 3 linear + 1 full × 10 циклов |
| chat_template | Qwen3 simple | Qwen3.6 multimodal-aware |
| `tie_word_embeddings` | True (force-set) | **False** (genesis уже без tie) |
| Untie trick | актуально (см. finetune3) | **неактуально** (уже untied by design) |
| win_margin | 0.02 | **0.06** (3× жёстче — `score_chal ≥ 0.53`) |
| Тренировочное время | ~30 мин | ~2-4 часа |

---

## 4. Пайплайн

### Шаг 1. Подготовить pod

Заказать **4× H100 80GB**, 300+ GB диск. На образе
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`:

```bash
# Скопировать workspace
scp -r /root/albedo_run user@pod:/root/

# Установить deps
bash /root/albedo_run/pod_install.sh
# + дополнительные для multi-GPU MoE:
pip install --quiet "accelerate>=0.34.0" "deepspeed>=0.15.0"
# vLLM для inference (если ещё нет):
pip install --quiet "vllm>=0.6.0"
```

### Шаг 2. Скачать genesis

```bash
cd /root/albedo_run/finetune4
source /root/albedo/.venv/bin/activate

# Полная скачка ~72 GB:
python3 scripts/download_genesis_v4.py \
  --out /workspace/models/qwen36_35b_genesis

# ИЛИ облегчённая (без vision) ~50 GB:
python3 scripts/download_genesis_v4.py \
  --out /workspace/models/qwen36_35b_genesis \
  --skip-vision
```

⚠️ `--skip-vision` экономит ~22 GB, но если validator проверяет, что vision тензоры
существуют — наша модель отвалится. До выкатки нового кода — **не использовать**.

### Шаг 3. Перетокенизировать датасет под Qwen3.6

```bash
python3 scripts/build_sft_v4.py \
  --source-jsonl /root/albedo_run/finetune2/data/sft/train_v3.jsonl \
  --tokenizer /workspace/models/qwen36_35b_genesis \
  --out data/sft/train_v4.jsonl \
  --max-tokens 8192
```

Что произойдёт:
- Парсим старый текст `<|im_start|>role\\n...<|im_end|>` обратно в messages
- Re-применяем `apply_chat_template` с Qwen3.6 tokenizer'ом
- Заново фильтруем injection через PROBE_RISK regex
- Записываем `text`, `prompt`, `completion`, `n_tokens` поля

Ожидаем ~2300-2400 examples (тот же объём что в train_v3.jsonl).

### Шаг 4. SFT тренировка (4 × H100, ZeRO-3)

```bash
cd /root/albedo_run/finetune4
tmux new-session -d -s train4 -c .
tmux send-keys -t train4 'source /root/albedo/.venv/bin/activate && \
accelerate launch \
  --num_processes 4 \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_zero3.json \
  scripts/train_v4.py \
  --base /workspace/models/qwen36_35b_genesis \
  --data data/sft/train_v4.jsonl \
  --output ckpt/v4 \
  --lr 1e-5 --max-steps 200 --save-steps 50 \
  --batch-size 1 --grad-accum 4 \
  --freeze-vision \
  2>&1 | tee /tmp/train_v4.log; echo "EXIT=${PIPESTATUS[0]}"' Enter

tail -f /tmp/train_v4.log | grep -E "'loss'|checkpoint|FAIL"
```

ETA: ~2-4 часа на 200 шагов.

⚠️ **Известные риски**:
1. **transformers 5.12.1 может не поддерживать Qwen3_5MoeForConditionalGeneration**
   → upgrade transformers, либо использовать AutoModel.from_pretrained()
2. **TRL completion_only_loss может не работать с multimodal**
   → fallback на standard SFT (поле `text`)
3. **flash-attention 2 для MoE с linear_attention** — может потребовать спец-флага
4. **ZeRO-3 + CPU offload** на больших chat-моделях — медленный, но работает
5. **Vision encoder freeze** — может не сработать если атрибут называется по-другому

### Шаг 5. Reshard в 35+ × 2 GB

```bash
python3 scripts/reshard_v4.py \
  --src ckpt/v4/final \
  --dst ckpt/v4/final_sharded \
  --max-shard 2GB
```

После reshard — ~35-40 шардов по 2 GB.

### Шаг 6. Локальная валидация

⚠️ TBD до выкатки нового chain.toml. После — обновить validate_local.py с новыми
ALL_LOCK_KEYS и repo_pattern, и запустить.

```bash
# TODO: создать validate_local_v4.py
```

### Шаг 7. Локальный eval через vLLM endpoints

Скачать текущего короля (когда новая компетиция начнётся):

```bash
python3 scripts/download_genesis_v4.py \
  --repo <king_namespace>/<king_repo> \
  --digest <king_digest> \
  --out /workspace/models/champion_v4
```

На pod'е запустить 2 vLLM сервера:
```bash
# Pod terminal 1 — king на GPU 0-1
CUDA_VISIBLE_DEVICES=0,1 vllm serve \
  /workspace/models/champion_v4 \
  --port 8001 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192

# Pod terminal 2 — challenger на GPU 2-3
CUDA_VISIBLE_DEVICES=2,3 vllm serve \
  /root/albedo_run/finetune4/ckpt/v4/final_sharded \
  --port 8002 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192
```

В третьем терминале — eval:
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python3 scripts/eval_local_v4.py \
  --king-url http://localhost:8001 \
  --chal-url http://localhost:8002 \
  --king-tokenizer /workspace/models/champion_v4 \
  --chal-tokenizer /root/albedo_run/finetune4/ckpt/v4/final_sharded \
  --n-samples 32 --max-turns 3 \
  --cache-dir /tmp/eval_v4_vs_champ
```

Ожидаем: верится с margin ≥ 0.06 (новый порог). При 32 sample CI ~±0.09.

### Шаг 8. Upload в Hippius (35+ × 2 GB)

```bash
# Те же env vars что в myalbedo.md §8.2
export HIPPIUS_UPLOAD_WORKERS=1 HIPPIUS_MAX_CONCURRENT=4 HIPPIUS_CHUNK_SIZE=33554432

# Repo pattern может измениться — узнать ПОСЛЕ выкатки нового chain.toml
# Предположение: ALBEDO_REPO_PREFIX=albedo-qwen3.6-35b (вместо albedo-qwen3-4b)
for t in 1 2 3 4 5 6; do
  hippius-hub upload tarab/albedo-qwen3.6-35b-v1 \
    /root/albedo_run/finetune4/ckpt/v4/final_sharded \
    --revision main && break
  sleep 5
done
```

### Шаг 9. Register + Commit v6

```bash
cd /root/albedo_repo  # или /root/albedo если переехало

# Новый хоткей (aff4)
albedo register --coldkey runp --hotkey aff4 --netuid 97

# Commit
albedo commit \
  --repo tarab/albedo-qwen3.6-35b-v1 \
  --digest sha256:<DIGEST> \
  --coldkey runp --hotkey aff4
```

---

## 5. Что точно знаем / что предполагаем

### Точно знаем (из дискорда + dashboard + конфига скачанного genesis)

✅ Новый genesis: `teutonic/qwen3.6-35b-a3b-genesis @ sha256:efd5b8d0…23e89c165`
✅ Архитектура: Qwen3_5MoeForConditionalGeneration (multimodal MoE)
✅ Размер: 35B params (3B active), 72 GB на диске
✅ tokenizer: vocab 248320 (другой!)
✅ tie_word_embeddings: False (genesis уже untied)
✅ win_margin: 0.06 (3× жёстче)
✅ Validator всё ещё ждёт only bf16/fp16 (quantization не пройдёт)
✅ Sanity-prompts те же: `double(x)`, `add(a,b)`, `print(x[5])`
✅ Judges те же: `z-ai/glm-5.1`, `qwen/qwen3.5-397b-a17b`, `deepseek/deepseek-v3.2`

### Предполагаем

⚠️ Новый repo_pattern: `^[^/]+/albedo-qwen3.6-35b-.+$` (вероятно)
⚠️ ALL_LOCK_KEYS будут охватывать MoE-specific поля (num_experts, ...)
⚠️ Validator адаптирует логику для multi-modal (либо проверять text_config, либо
   добавить root-level mirroring)
⚠️ Vision encoder обязательная часть модели (без него может не пройти)
⚠️ flash-attention 2 нужен для linear_attention (Mamba-style state space)

### Открытые вопросы

❓ Можно ли заморозить vision encoder без нарушения validator'a?
❓ Поддерживает ли HF transformers v5.12.1 модель `qwen3_5_moe`?
❓ Поддерживает ли vLLM генерацию на этой модели?
❓ Достаточно ли train_v3.jsonl (2300 examples) для 35B модели?
   (рекомендация ML practice: > 5K для серьёзного SFT 35B+)
❓ Лучше ли LoRA вместо full SFT для clone-with-improvement (новый король 35B
   с LoRA-trickle — может быть оптимально по cost)?

---

## 6. Бюджет

| Фаза | Стоимость |
|---|---|
| GPU rent (4× H100 на 6 ч) | ~$80-150 |
| Hippius credits для namespace (если новый нужен) | ~$15-25 |
| TAO для регистрации хоткея | ~τ2-3 = $400-500 |
| **Итого минимум 1 итерация** | **~$500-700** |

Для 2-3 итераций v4 → v5 → v6 — ~$1500-2000.

---

## 7. Чек-лист готовности (заполняем по мере выкатки)

- [ ] chain.toml на main обновлён под Qwen3.6-35B-A3B
- [ ] architecture.py обновлён под multimodal lock keys
- [ ] win_margin 0.06 точно в проде (был на dev)
- [ ] transformers v5.x поддерживает `qwen3_5_moe`
- [ ] vLLM поддерживает `Qwen3_5MoeForConditionalGeneration`
- [ ] Скачан genesis (72 GB) на pod
- [ ] build_sft_v4.py создал train_v4.jsonl без ошибок
- [ ] train_v4.py запустился без OOM на 4× H100
- [ ] reshard_v4.py выдал 35+ × 2 GB шардов
- [ ] fingerprint_check.py показывает sim < 0.95 vs all references
- [ ] Hippius upload прошёл retry-loop без 404
- [ ] albedo check-hippius VALID на нашей залитой модели
- [ ] Хоткей зарегистрирован, баланс ≥ recycle
- [ ] **ТОЛЬКО ПОСЛЕ ВСЕГО** — albedo commit

---

## 8. План минимального действия (если ресурсы ограничены)

Не делать full SFT — попробовать **LoRA от чемпиона** через PEFT:

```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# trainable params ~50M вместо 35B
```

Преимущества:
- Можно запустить даже на 1× H100 80GB
- Тренировка 30 мин вместо 4 ч
- Меньше риск сломать MoE routing
- При merge — лёгкое отклонение для прохождения fingerprint dedup

Но: TRL + LoRA + multimodal MoE — untested combo. Готовьтесь к багам.

Этот вариант **не покрыт** train_v4.py — нужно отдельный `train_v4_lora.py`.
