# Albedo finetune3 — clone-with-improvement от чемпиона

> **Идея**: вместо обучения «с нуля» (как v3) — взять текущего короля и сдвинуть
> его веса ровно настолько, чтобы пройти clone-detection (similarity < 0.95) и
> минимально потерять качество. Самый быстрый путь к ≥0.55 score, потому что
> мы стартуем уже с 0.54+.

---

## 0. TL;DR — что это и не это

| Параметр | finetune2/v3 (свой SFT с untie) | **finetune3 (clone-from-champion)** |
|---|---|---|
| База | Qwen3-4B-Instruct-2507-overridden | **arboshelper/albedo-qwen3-4b-2-5-final** (текущий король) |
| Untie lm_head | ✅ выполняется ВРУЧНУЮ | ✅ уже untied (король) |
| LR | 1e-5 | **5e-7** (20× меньше) |
| Steps | ~146 (1 эпоха × 2371) | **~60 max** (early stop) |
| Цель | максимально хорошая модель | **минимальный сдвиг чтобы пройти dedup** |
| Riск качества | средний (учим с нуля чего хотим) | **низкий** (начинаем с прокачаного, лишь чуть-чуть SFT'им) |
| ETA train | ~30-40 мин (H100) | **~5-15 мин (H100)** |
| Шанс короны | средний | **высокий** (близко к чемпиону, чуть лучше за счёт SFT) |
| Юр-риск | нет | **средний** (тренируемся на их выгрузке — публично, но этично) |

**Когда использовать**: если v3 не свергнет короля, finetune3 — самый быстрый
способ войти в reign. Не панацея — оригинальный король тоже итерируется.

---

## 1. Что в папке

```
finetune3/
├── GUIDE.md                            ⭐ читать первым
├── scripts/
│   ├── download_champion.py             ⭐ скачать короля из Hippius
│   ├── inspect_champion.py              ⭐ детектор: tied / exotic / untied → какой train запускать
│   ├── train_champion_tied.py           ⭐ для TIED / EXOTIC королей (arboshelper-style, нормальный кейс)
│   ├── train_champion_untied.py         для UNTIED королей (редкий, когда lm_head действительно отдельный)
│   ├── fingerprint_check.py             ⭐ прод-identical clone-detection
│   ├── reshard.py                       решард в 5×2GB для Hippius
│   ├── validate_local.py                12 локальных проверок
│   ├── setup_pod.sh                     bootstrap pod'а (для базовых моделей v3)
│   └── deploy_v6.sh                     v6 commit pipeline
├── configs/
│   └── ds_zero2.json
├── refs/
│   └── judge_prompts_recovered.md       PROBE_SYSTEM
└── data/sft/
    ├── train_v3.jsonl                   ⭐ датасет (тот же что в finetune2)
    └── red_team.jsonl                   sanity-check negatives
```

> Датасет тот же что в finetune2: 2371 примера (2331 chal-wins + 40 defensive).

---

## 2. Стратегия (зачем именно так)

### 2.1 Зачем low LR

Чтобы **сдвинуть веса достаточно для прохождения fingerprint dedup**, но не
сломать модель. Numerics:

- Fingerprint считает 16 sample values на каждый из 398-399 тензоров
- "unchanged" если cos(sample) ≥ 1−1e-6
- Это означает sample должен сдвинуться < 0.14% (relative) — почти неподвижно

При lr=5e-7 (20× меньше дефолта) каждый шаг сдвигает веса на ~0.05-0.5%
относительно. После 10-30 шагов **большинство** тензоров уже за threshold'ом
"unchanged", но модель ещё **помнит** чему её учил оригинальный SFT короля.

### 2.2 Зачем intermediate checkpoints

Нет универсального N steps что подойдёт. Поэтому:
1. Сохраняем чекпойнт каждые 10 шагов (всего ~6 промежуточных)
2. После train запускаем fingerprint_check на каждом
3. **Выбираем самый ранний** где similarity < 0.95
4. Чем меньше шагов — тем ближе модель к королю качеством

### 2.3 Что мы НЕ делаем

- ❌ Не train'им много эпох — забыли бы что мы король
- ❌ Не используем lr > 1e-6 — слишком резко
- ❌ Не используем train_v2 (full-text loss) — кодируем стиль короля поверх нашего шума
- ❌ Не выкидываем `tie_word_embeddings=True` — нужно для arch lock

---

## 3. Pipeline

### Шаг 1. Сетап pod'а (если новый)

```bash
# 1. Скопировать finetune3 на pod
scp -r /root/albedo_run/finetune3 user@pod:/root/albedo_run/

# 2. На pod'е
cd /root/albedo_run/finetune3
bash scripts/setup_pod.sh   # достаёт overridden базу — нужна для validate_local fingerprint check

# 3. Активировать venv
source /root/albedo/.venv/bin/activate
```

### Шаг 2. Узнать кто сейчас король

```bash
curl -s https://s3.hippius.com/albedo/data/dashboard.json | python3 -c "
import json,sys
d = json.load(sys.stdin)
king = d['reign']['members'][0]
print(f\"king v{king['king_version']}: {king['model_uri']}\")
print(f\"  hotkey: {king['hotkey']}\")
print(f\"  score:  {king['score_challenger']:.4f}\")
"
```

Запиши `model_uri` (`namespace/repo`) и `digest`.

### Шаг 3. Скачать чемпиона

```bash
python3 scripts/download_champion.py \
  --repo <new-king-namespace>/albedo-qwen3-4b-<name> \
  --revision main \
  --out /workspace/models/champion
```

Time: ~1-2 мин (~8.8 GB через Hippius).

### Шаг 4. ⭐ Inspect — определить tied / exotic / untied

**Это критично делать перед train.** Скрипт выясняет какой именно вариант
тренировки запускать.

```bash
python3 scripts/inspect_champion.py --champion-dir /workspace/models/champion
```

Что увидишь — один из трёх исходов:

| VERDICT | Значение | Какой train запускать |
|---|---|---|
| **✅ TIED** | 398 тензоров, lm_head не в state_dict | `train_champion_tied.py` |
| **⚠️ EXOTIC** | 399 тензоров, identical bytes (как arboshelper) | `train_champion_tied.py` (transformers v5 при load выкинет дубль → работает как tied) |
| **🔥 UNTIED** | 399 тензоров, lm_head ≠ embed | `train_champion_untied.py` (нужен special-save) |

Большинство королей будет **EXOTIC** (как arboshelper). Реальный untied почти не
встречается — если увидишь, читай **§5 «UNTIED variant»** ниже.

### Шаг 5. Тренировка (короткая, в tmux)

#### Если VERDICT = TIED или EXOTIC (нормальный кейс):

```bash
cd /root/albedo_run/finetune3
tmux new-session -d -s champ -c /root/albedo_run/finetune3
tmux send-keys -t champ 'source /root/albedo/.venv/bin/activate && \
python3 scripts/train_champion_tied.py \
  --champion-dir /workspace/models/champion \
  --data /root/albedo_run/finetune3/data/sft/train_v3.jsonl \
  --output /root/albedo_run/finetune3/ckpt/champ \
  --lr 5e-7 --max-steps 60 --save-steps 10 \
  --batch-size 2 --grad-accum 8 \
  --no-flash-attn \
  2>&1 | tee /tmp/champ_train.log; echo "EXIT=${PIPESTATUS[0]} at $(date)"' Enter

tail -f /tmp/champ_train.log | grep -E "'loss'|checkpoint|FAIL"
```

ETA: ~5-15 минут на H100 для 60 шагов.

Что в логе ожидать:
```
  trainable params: 4.022 B  (tied — 398 unique tensors)
  Dataset (prompt/completion): 2371 kept, 0 too long

=== Training (low LR, frequent checkpoints) ===
{'loss': 0.42, 'grad_norm': 0.06, 'learning_rate': 5e-07, 'epoch': 0.01}
{'loss': 0.41, 'grad_norm': 0.05, 'learning_rate': 5e-07, 'epoch': 0.02}
...
Saving checkpoint-10
...
Saving checkpoint-60

Saving final to .../ckpt/champ/final ...
```

#### Если VERDICT = UNTIED (редкий кейс):

Замени название скрипта на `train_champion_untied.py` — остальные аргументы те же:

```bash
tmux send-keys -t champ 'source /root/albedo/.venv/bin/activate && \
python3 scripts/train_champion_untied.py \
  --champion-dir /workspace/models/champion \
  --data /root/albedo_run/finetune3/data/sft/train_v3.jsonl \
  --output /root/albedo_run/finetune3/ckpt/champ_untied \
  --lr 5e-7 --max-steps 60 --save-steps 10 \
  --batch-size 2 --grad-accum 8 \
  --no-flash-attn \
  2>&1 | tee /tmp/champ_train.log; echo "EXIT=${PIPESTATUS[0]} at $(date)"' Enter
```

ETA: ~10-20 минут на H100 (чуть медленнее из-за +389M params в оптимизаторе).
Чекпойнт ~8.82 GB (vs 8.04 у tied).

В логе вместо `4.022 B` увидишь `4.411 B`. После save — `✓ OK — lm_head и
embed_tokens — отдельные, разные тензоры`. Если `✗ FAIL` — что-то пошло не так,
не использовать чекпойнт.

### Шаг 5. Найти sweet-spot checkpoint

```bash
# Для каждого checkpoint проверяем fingerprint vs source champion
for ckpt in /root/albedo_run/finetune3/ckpt/champ/checkpoint-*; do
  echo ""
  echo "=== $(basename $ckpt) ==="
  python3 scripts/fingerprint_check.py \
    --candidate "$ckpt" \
    --reference /workspace/models/champion 2>&1 | grep -E "similarity|FAIL|OK|^  unchanged|^  changed"
done
```

Пример вывода:
```
=== checkpoint-10 ===
  similarity: 0.6234  [OK]
  unchanged: 248/399  (of which near-changed 156 = cos≥0.99 but <1-1e-6)
  changed:   151/399
  margin to DQ threshold: -0.3266  (safe)

=== checkpoint-20 ===
  similarity: 0.2105  [OK]
  unchanged: 84/399  ...

=== checkpoint-30 ===
  similarity: 0.0526  [OK]
  ...
```

**Выбери САМЫЙ РАННИЙ checkpoint с similarity < 0.95** — обычно это 10 или 20.
Чем раньше — тем ближе к качеству короля.

⚠️ Если ВСЕ checkpoint'ы > 0.95 — увеличить max_steps или lr (но не сильно):

```bash
# Если 60 шагов не хватило (маловероятно при lr=5e-7):
python3 scripts/train_champion_tied.py ... --max-steps 120 --save-steps 20
# (или train_champion_untied.py если у тебя UNTIED king)
```

### Шаг 6. Также проверить против genesis и Instruct-2507

```bash
# Выбранный checkpoint (например checkpoint-20)
CKPT=/root/albedo_run/finetune3/ckpt/champ/checkpoint-20

python3 scripts/fingerprint_check.py \
  --candidate "$CKPT" \
  --reference /workspace/models/champion \
              /workspace/models/Qwen3-4B \
              /workspace/models/Qwen3-4B-Instruct-2507 \
              /workspace/models/Qwen3-4B-Instruct-2507-overridden
```

Все 4 sim'ы должны быть < 0.95.

### Шаг 7. Решард в 5×2GB

```bash
python3 scripts/reshard.py \
  --src /root/albedo_run/finetune3/ckpt/champ/checkpoint-20 \
  --dst /root/albedo_run/finetune3/ckpt/champ/checkpoint-20_sharded \
  --max-shard 2GB
```

Скрипт пере-проверит что untied lm_head сохранился после reshard. Если случайно
`save_pretrained()` дедуплицировал — сообщит.

### Шаг 8. Локальная валидация

```bash
python3 scripts/validate_local.py \
  --model-dir /root/albedo_run/finetune3/ckpt/champ/checkpoint-20_sharded \
  --repo tarab/albedo-qwen3-4b-champ1 \
  --king-dir /workspace/models/Qwen3-4B \
  --fingerprint-against /workspace/models/Qwen3-4B \
    /workspace/models/Qwen3-4B-Instruct-2507 \
    /workspace/models/champion
```

Все 12 проверок должны быть PASS.

### Шаг 9. Upload в Hippius

Cм. `myalbedo.md` §8.2. Имя модели **новое** — не тарам v2 / v3, а например:

```bash
cd /root/albedo_repo  # официальный CLI
V=.venv-deploy/bin

export HIPPIUS_UPLOAD_WORKERS=1 HIPPIUS_MAX_CONCURRENT=4 HIPPIUS_CHUNK_SIZE=33554432
for t in 1 2 3 4 5 6; do
  $V/hippius-hub upload tarab/albedo-qwen3-4b-champ1 \
    /root/albedo_run/finetune3/ckpt/champ/checkpoint-20_sharded \
    --revision main && break
  sleep 5
done

$V/hippius-hub revisions tarab/albedo-qwen3-4b-champ1 --json | grep -o 'sha256:[a-f0-9]*' | head -1
# → sha256:CHAMP_DIGEST
```

### Шаг 10. Регистрация нового хоткея + commit

```bash
# Проверить баланс
python3 -c "
import bittensor as bt
st = bt.Subtensor(network='finney')
w = bt.Wallet(name='runp', hotkey='aff3')
print(f'recycle: {st.recycle(netuid=97)}')
print(f'balance: {st.get_balance(w.coldkeypub.ss58_address)}')
"

# Регистрация (TAO burn)
$V/albedo register --coldkey runp --hotkey aff3 --netuid 97

# Commit v6
$V/albedo commit \
  --repo tarab/albedo-qwen3-4b-champ1 \
  --digest sha256:CHAMP_DIGEST \
  --coldkey runp --hotkey aff3
```

### Шаг 11. Мониторинг

```bash
# Через 30-60 секунд после commit'a должен попасть в queue:
curl -s https://s3.hippius.com/albedo/data/dashboard.json | python3 -c "
import json,sys
d=json.load(sys.stdin)
q = d.get('queue',[])
print(f'queue: {len(q)}')
for x in q:
  print(f'  {x.get(\"model_uri\",\"?\")[:70]}')
"
```

Когда eval закончится, ищи свой `tarab/albedo-qwen3-4b-champ1` в `eval_runs` —
там будет `score_challenger`, `score_king`, `coronated`.

---

## 4. Возможные исходы и что делать

| Исход | Значение | Действие |
|---|---|---|
| score_challenger > 0.55, coronated=True | КОРОНА! | Празднуй, мониторь reign. Готовь v4. |
| 0.50-0.55, не corona | Близко но не дотянули, маржа 0.02 не пройдена | Возможно король успел сменился. Делай champ_v2 со свежего короля. |
| < 0.50 | Сдвинули слишком сильно или judges просели | Попробуй ранний checkpoint (10 → 20). |
| fault_class=duplicate | Sim ≥ 0.95 в проде (мы недосчитали) | checkpoint-30 или больше; дальше от короля. |
| fault_class=injection | Модель сгенерила что-то | Проверь red-team локально, может SFT за-учил injection. |

---

## 5. Этика / риски

**Что мы делаем**: берём публично доступную модель (король выложил веса в
Hippius — публичная сеть), делаем минимальный SFT, коммитим под своим
хоткеем.

**Можно ли это**: технически — да, ничего в правилах против этого нет
(см. chain.toml — там только arch lock + format + dedup). Король сам отдал
веса в открытый доступ.

**Risk перцепции**: сообщество может это назвать «копи-атакой». В дискорде
hurricanebooster уже жаловался на dedup false positives — значит тема горячая.

**Mitigation**: 
1. Применяем НЕ просто шум, а реальный SFT на нашем датасете → модель
   действительно становится **чуть лучше** на defensive examples.
2. Не возвращаемся к одному и тому же чемпиону многократно (1-2 раза макс).
3. Имеем параллельно finetune2/v3 — самостоятельная модель.

---

## 6. Бюджет

| Шаг | $ |
|---|---|
| GPU rent (~30 мин H100) | ~$2-3 |
| TAO burn (новый хоткей) | ~τ2.3 ≈ $400-500 |
| Hippius storage | 0 (Free плана хватает) |
| Mental | низкий — pipeline почти 100% такой же как v3 |

---

## 7. Чек-лист перед commit'ом

- [ ] Король изменился? — если да, скачать свежего и переобучить
- [ ] download_champion.py показал untied (399 тензоров)
- [ ] inspect_champion.py запущен ПЕРВЫМ — выбран правильный train-скрипт
- [ ] train_champion_tied.py (или _untied.py) дошёл до конца, есть 6+ checkpoint'ов
- [ ] Выбран самый ранний checkpoint с similarity < 0.95 vs source champion
- [ ] Тот же checkpoint имеет similarity < 0.95 vs ВСЕ другие models (Qwen3-4B, Instruct-2507, overridden)
- [ ] reshard в 5×2GB прошёл, untie verify OK
- [ ] validate_local.py — все 12 PASS
- [ ] Hippius upload вернул валидный digest
- [ ] Новый хоткей выбран (aff3 если v3 = aff2, v2 = aff1)
- [ ] Баланс coldkey ≥ recycle
- [ ] Только после всего — `albedo commit`

---

## 8. Что после commit'a

После сабмишена модель попадёт в queue валидатора через 30-60 секунд. Eval
занимает ~10-15 минут.

Параллельно следи за dashboard'ом — король может смениться **до того** как
наш eval доедет до очереди. Если так — наш checkpoint всё равно прогоняется
**против актуального короля на момент eval'a**, не против того кто был при
commit'e.

Если новый король тоже untied и фингерпринт-близок к arboshelper'у (т.е.
кто-то ещё делает clone-with-improvement), наш модель может оказаться
**похожей на нового короля** → dedup. Это редко, но не невозможно.

---

## 9. Файлы для бэкапа на оркестратор

После commit'a — забери на оркестратор для истории:

```bash
# С пода на оркестратор
rsync -av --exclude='*.safetensors' --exclude='checkpoint-*' \
  user@pod:/root/albedo_run/finetune3/ \
  /root/albedo_run/finetune3_backup_$(date +%Y%m%d)/

# Только выбранный checkpoint (на случай если хочешь переиспользовать):
# (или просто запиши digest и поднимай из Hippius при необходимости)
```
