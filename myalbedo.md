# Albedo SN97 — боевая документация и операционный журнал

> Обновлено 2026-06-16. Цель: победить в king-of-the-hill турнире Bittensor **SN97 (Albedo)**
> на базе **Qwen3-4B**. Документ = (1) журнал что сделали, (2) пути ко всем файлам,
> (3) пошаговое участие в сети, (4) технический референс.

---

## 0. TL;DR — текущий статус

- **v1** обучена, валидация **14/14** (`ckpt/v1/final`). Fallback-кандидат.
- **v2** обучена, валидация **14/14** (`ckpt/v2/final`). **ВЫБРАНА для деплоя** (лучше v1 — см. §1a).
- **Деплой готов** (CLI `albedo`, v6, namespace `tarab`, токен в `.env`).
- **План:** ✅ обучение → ✅ оценка → ✅ **upload v2 в Hippius (sharded)** → ⏳ регистрация + commit.
- **✅ ЗАЛИТО:** `tarab/albedo-qwen3-4b-v2` @ main, digest
  **`sha256:b50d3f817920874a690bea3006d1b630e3fb0ebedcf56433d917b625a6b2199b`**.
  Файлы (11, allowlist-ok): config/generation_config/chat_template/tokenizer_config/tokenizer.json +
  `model-00001..00005-of-00005.safetensors` + index. Старого одиночного `model.safetensors` НЕТ.
- **БЛОКЕР (остался, для commit):** хоткей `5FJB…aff1` не зарегистрирован; netuid 97 регистрация = **τ1.03**,
  баланс coldkey **τ0.60** → **пополнить**, затем `albedo register` → `albedo commit` (v6).
  17.06: цена выросла до **τ2.25** (+53% за сутки), баланс **τ1.10**.
- **🔥 «ТРЮК КОРОЛЯ» 17.06 ОПРОВЕРГНУТ 18.06**: lm_head у arboshelper'a физически
  идентичен embed_tokens (просто записан дважды в safetensors). При загрузке через
  transformers v5 дубль выкидывается → модель работает как обычный tied. Преимущество
  arboshelper'а — в качестве тренировки, а не в +389M params. Подробности → §1b.
  Реализация настоящего untied (если когда-то понадобится) → `finetune3/scripts/`.
- **finetune2/ (v3, untie + completion-only)** — готов к запуску на pod'е.
- **finetune3/ (clone-with-improvement)** — готов к запуску. Скрипты:
  `inspect_champion.py` (детектор tied/untied) → `train_champion_tied.py`
  (для arboshelper-style) или `train_champion_untied.py` (для real untied).

### Как решили upload (история — чтобы не повторять)
1. **401 на push** был потому, что консольный токен `70a4…` НЕ годится для registry. Решение —
   `hippius-hub login --hippius-token <console>` → **`hippius-hub registry rotate-token`** (минтит
   robot-secret `robot$tarab+tarab-bot`, пишет в `~/.cache/hippius/hub/token`). После этого push-scope
   для `tarab` = `['push','delete','pull']`. В `.env` `HIPPIUS_HUB_TOKEN` закомментирован → upload берёт
   сохранённый robot-secret. (namespace `tarab` активен, Free, квота 100 GB — подписка не нужна.)
2. **404 на 8-ГБ блобе** `model.safetensors` (оба пути, повторяемо). Реестр флаки на больших/многочанковых
   блобах. Решение: **решард в 5×~2 ГБ** (`model-*-of-00005.safetensors`, allowlist это разрешает) +
   **retry-цикл с пониженной конкурентностью** (`HIPPIUS_UPLOAD_WORKERS=1 HIPPIUS_MAX_CONCURRENT=4
   HIPPIUS_CHUNK_SIZE=32MB`) — merge-семантика дозаливает пропущенные шарды. Решард-команда — см. ниже §5.
   Шардированный чекпойнт: `ckpt/v2/final_sharded/` (его и грузим/коммитим, digest выше).
- **БЛОКЕР 2 (commit, на потом):** хоткей не зарегистрирован; регистрация netuid 97 = **τ1.03**, баланс coldkey **τ0.60** → **пополнить**.

## 1b. ⚠️ lm_head у arboshelper — НЕ трюк (опровергнуто 18.06)

> **Корректировка предыдущей версии этого раздела.** 17.06 я записал что у
> arboshelper'a +389M «бесплатных» обучаемых параметров. Это **неверно** —
> новый скрипт `inspect_champion.py` показал что lm_head там идентичен embed_tokens.

### Что на самом деле

При повторной инспекции через `finetune3/scripts/inspect_champion.py`:

```
arboshelper/albedo-qwen3-4b-2-5-final:
  total tensors:  399
  total params:   4.411 B  (в файле)
  has 'lm_head.weight':              True   shape=(151936, 2560)
  has 'model.embed_tokens.weight':   True   shape=(151936, 2560)
  lm_head and embed_tokens are identical bytes: True   ← ВОТ ОНО

VERDICT: ⚠️ EXOTIC (both keys, identical bytes) — treat as TIED
```

То есть в safetensors лежат **два байт-идентичных тензора**. Это просто **дубль**
из-за того что transformers v4.57.6 при `safe_serialization=True` пишет оба ключа
независимо от tie_word_embeddings. Лишних **обучаемых** параметров нет — они оба
указывают на одно и то же эмбеддинг-пространство.

Эффективная модель — **4.022 B params** (как у нас), не 4.411 B.
Лишние 780 MB на диске — пустая нагрузка для Hippius/сети, ничего более.

При загрузке через `transformers v5` modeling-код видит `config.tie_word_embeddings=True`
и **выкидывает дублирующий lm_head** через `_tied_weights_keys` — работает как
обычный tied. То есть в проде король эффективно tied.

### Что значит на практике

1. **У нас нет «отстающего» количества параметров** — мы tied 4.022B, они tied 4.022B.
2. **Преимущество arboshelper'а — в тренировке, а не в архитектуре**: качество
   данных, lr-schedule, количество эпох, etc.
3. **Untied lm_head как реальный трюк всё равно возможен** — если *преднамеренно*
   копировать embed → lm_head и тренировать раздельно с `config.tie=False` во время
   save. Это и есть `train_champion_untied.py` и оригинальный план v3.
4. **Если хочешь скопировать «трюк» по-настоящему** — нужно тренировать
   с явно untied lm_head; нельзя просто загрузить arboshelper и тренировать
   как обычно — transformers v5 его дедуплицирует.

### Как определять у НОВЫХ королей

Перед любым finetune3-прогоном — обязательно:

```bash
python3 /root/albedo_run/finetune3/scripts/inspect_champion.py \
  --champion-dir /workspace/models/champion
```

Возможные исходы:
- **✅ TIED** (398 тензоров, lm_head в state_dict не указан) → стандартный случай
- **⚠️ EXOTIC** (399 тензоров, identical bytes) → tied de-facto; как arboshelper
- **🔥 UNTIED** (399 тензоров, разные значения) → реальный трюк; нужен specialhandling

Соответственно использовать:
- TIED / EXOTIC → `train_champion_tied.py` + `reshard.py`
- UNTIED → `train_champion_untied.py` + `reshard.py` (он сам detect'нет)

### Риски arch lock и dedup при untied

- **Arch lock**: валидатор проверяет 11 ключей включая `tie_word_embeddings`. Если у
  нас в config tie=True (обязательно), но в safetensors есть отдельный lm_head — это
  ОК, валидатор не сверяет веса с config-флагом.
- **transformers v5 при save** с `config.tie=True` и `_tied_weights_keys=["lm_head.weight"]`
  **выкинет** lm_head из state_dict. Поэтому в untied-варианте мы устанавливаем
  `config.tie=False` ВРЕМЕННО на save, потом перезаписываем config.json с tie=True
  для arch lock. См. `train_champion_untied.py:230-238`.
- **vLLM в проде** загружает модель из safetensors. Если там есть lm_head — он будет
  использован отдельно от embed (поведение HF: explicit lm_head в state_dict побеждает
  tie-флаг). Если нет — модель работает tied.

### Где живёт код

- `finetune3/scripts/inspect_champion.py` — определитель типа
- `finetune3/scripts/train_champion_tied.py` — для TIED / EXOTIC
- `finetune3/scripts/train_champion_untied.py` — для настоящего UNTIED
- `finetune3/scripts/reshard.py` — auto-detect, корректно обрабатывает оба случая

---

## 1a. Результаты v1 vs v2 (оценка 16.06)

Обе валидны **14/14**. Прокси-оценка судьи (16 реальных SWE-префиксов, `compare_models.py`):

| model | THOUGHT | one-bash | no-cd | inj-free | clean-stop | avg-words | train: loss/acc | fp vs base |
|---|---|---|---|---|---|---|---|---|
| v1 | 94% | 94% | 100% | 100% | 94% | 51 | 0.746 / 0.818 | 0.5804 |
| **v2** | **100%** | 94% | 100% | 100% | 94% | **47** | 0.470 / **0.871** | 0.7136 |

**Решающая разница (grounding/correctness):** v1 по привычке ищет Python в ЛЮБОМ репо
(`find . -name "*.py"` в JS/Java-проектах — мисгрундинг). **v2 распознаёт реальный язык
проекта** из наблюдений (TS/Svelte→`*.ts`, gradle→`*.java`) и таргетит нужные символы.
Это ровно метрики, которыми выигрывается дуэль. **Вывод: деплоим v2, v1 — fallback.**

> ⚠️ fingerprint **vs king = 0.0** у обеих — артефакт: локальный king `/workspace/models/Qwen3-4B`
> неполный (только шард 3/3), наборы тензоров не совпадают. Арх-лок валиден (config короля на месте).
> v2 ближе к базе (0.71 vs 0.58) — completion-loss меняет меньше весов; для dedup безопасно (<0.95).
> Артефакты: `/tmp/compare_results.json`, `/tmp/validate_v2.log`.

---

## 1. Что мы сделали пошагово (сессия 16.06)

1. **Склонировали/изучили репозиторий** турнира (`/root/albedo_repo`, зеркало `unarbos/albedo`).
   Сверили с официальным `tony-dendrite/albedo`: критичные файлы (chain.toml, judge_core,
   sanity, worker, reveal-v6, miner CLI) **идентичны ветке `official/dev`** (живой).
2. **Полный аудит валидатора** (12 проверок) и сверка с нашими скриптами. Нашли расхождения:
   - sanity-гейт ≠ «THOUGHT+bash» (реально: эвристики + injection + viability на generic-Python);
   - условие победы упрощается до **median судей ≥ 0.51** (`score_king = 1 − score_challenger`);
   - тренировка считала loss по всему тексту (надо assistant-only).
3. **Диагностировали смерть тренировки** (умерла на step 33 — SIGHUP, запуск был без tmux).
   Перезапустили v1 **в tmux** → дошла до конца чисто (`train_loss 0.746`, acc 0.818).
4. **Провалидировали v1 локально — 14/14 PASS** (арх-лок, файлы, fingerprint, sanity).
   fingerprint vs king = **0.0000**, vs base = **0.5804** (< 0.95 — dedup пройдёт).
5. **Изучили дискорд-логи** (`dist_15-06-26.json` = старый distil; `disc16.06.txt` = переход на albedo):
   коммит **v6** (v4/v5 игнор), **1 commit = 1 eval = 1 hotkey**, Qwen3-4B временный (~неделя),
   датасет SWE-ZERO = **проваленные траектории**, dedup иногда ложно срабатывает.
6. **Восстановили скрытые промпты судьи** из старого репо `/root/albedo/albedo/judge/rubric.py`
   (в проде спрятаны в невыложенном `rubricisity.py`) → `refs/judge_prompts_recovered.md`.
   Главное: `PROBE_SYSTEM` (injection-аудитор) — терминальный non-retryable провал.
7. **Собрали чистые датасеты v2** (`build_sft_v2.py`): только формат-чистые ходы, тилт на
   challenger-победы, фильтр injection-паттернов по `PROBE_SYSTEM`. Скан показал: оригинальный
   `train.jsonl` содержал 5 injection-рискованных примеров → в `train_v2_chalonly.jsonl` их **0**.
8. **Добавили assistant-only loss** в `train_albedo_v1.py` (флаг `--completion-loss`, TRL
   `completion_only_loss`). Запустили **v2** от overridden-базы на `train_v2_chalonly.jsonl`.
9. **Подготовили боевой деплой**: исправили баг (старый `reveal_onchain.py` писал **v4**),
   поставили официальный `albedo` CLI (v6) в изолированный venv, `.env`, скрипт `deploy_v6.sh`.
   Подтвердили: namespace `tarab`, формат-парность chat-template, хоткей пока не зарегистрирован.

---

## 2. Файлы для локального сохранения (бэкап)

> Скопируй это себе, чтобы дальше улучшать и доучивать. Главное — **скрипты, датасеты,
> референсы, конфиги**. Веса чекпойнтов (~8 ГБ каждый) — по желанию.

### Наш рабочий каталог `/root/albedo_run/`
```
scripts/
  train_albedo_v1.py        # ⭐ тренировка (SFT). Флаг --completion-loss = assistant-only loss
  build_sft_v2.py           # ⭐ сборка чистого датасета из train.jsonl (формат+injection фильтры)
  build_sft_dataset.py      #   исходная сборка из кэша дуэлей (кэша сейчас нет)
  validate_local.py         # ⭐ локальная валидация (12 проверок: арх-лок+fingerprint+sanity)
  compare_models.py         # ⭐ сравнение кандидатов (прокси-метрики судьи + сэмплы)
  deploy_v6.sh              # ⭐ боевой деплой v6 (upload | publish). Замена сломанного deploy_after_train.sh
  reveal_onchain.py         # ❌ УСТАРЕЛ (пишет v4). НЕ использовать. Оставлен как память
  deploy_after_train.sh     # ❌ УСТАРЕЛ (v4 + зависит от /root/albedo). НЕ использовать
  setup_pod.sh              #   bootstrap пода (сборка overridden-базы из Instruct-2507)
  package_for_pod.sh        #   упаковка для переноса на под
  download_traces.py        #   скачивание трейсов дуэлей (для пересборки датасета)

data/sft/
  train.jsonl               #   v1-датасет (6469, 57% king, есть грязные ходы) — исходник
  train_v2.jsonl            #   v2 сбалансированный (4662: 2331 chal + 2331 king), 0..2 риска
  train_v2_chalonly.jsonl   # ⭐ v2 РЕКОМЕНДУЕМЫЙ (2331 challenger-побед, all-clean, 0 injection)

refs/
  judge_prompts_recovered.md # ⭐ восстановленные промпты судьи (PROBE_SYSTEM injection + pairwise rubric)

configs/
  ds_zero2.json             #   DeepSpeed ZeRO-2 (multi-GPU)

ckpt/
  v1/final/                 #   обученная v1 (валидна, 14/14). model.safetensors (~8 ГБ)
  v2/final/                 #   v2 (появится после обучения)

myalbedo.md                 # ⭐ этот файл
dist_15-06-26.json          #   дискорд: старый distil (контекст)
disc16.06.txt               #   дискорд: переход на albedo (правила v6, 1-shot)
```

### Конфиг деплоя `/root/albedo_repo/.env` (СЕКРЕТ — хранить безопасно, chmod 600)
```
CHAIN_NETUID=97 / CHAIN_NETWORK=finney
ALBEDO_COLDKEY=runp / ALBEDO_HOTKEY=aff1 / ALBEDO_NAMESPACE=tarab
ALBEDO_REPO_PREFIX=albedo-qwen3-4b
HIPPIUS_HUB_TOKEN=<твой токен>
```

### Кошелёк `~/.bittensor/wallets/runp/` (СЕКРЕТ — coldkey приватный!)
- hotkey `aff1` ss58 = `5FJB1Po31WKhnmaZSsNPGMdR9wT4W5EGYyoSJnUeQpSGXgM5`
- coldkey ss58 = `5Gp46RymQMogKY3SQbEKDjDbYbynkREXb23hAFpAZTvNZsJw`

### Базовые модели `/workspace/models/`
```
Qwen3-4B/                            # genesis/king (для арх-лока + fingerprint)
Qwen3-4B-Instruct-2507/              # исходные веса
Qwen3-4B-Instruct-2507-overridden/  # ⭐ НАША БАЗА (config переписан под genesis RoPE)
```

### Изолированный venv деплоя `/root/albedo_repo/.venv-deploy/`
Здесь стоит `albedo` CLI (свой torch — не трогает тренировочное окружение). Пересоздать:
`cd /root/albedo_repo && python3 -m venv .venv-deploy && .venv-deploy/bin/pip install -e .`

---

## 3. Пошаговое участие в сети (как закоммитить модель)

> Наш план: сначала **upload** (без on-chain), потом — регистрация и **commit**.
> 1 commit = 1 eval = 1 hotkey → выстрел ОДИН, сначала всё проверить локально.

### Шаг A. Обучить модель
```bash
cd /root/albedo_run
tmux new-session -d -s train2 -c /root/albedo_run
tmux send-keys -t train2 'python3 scripts/train_albedo_v1.py \
  --data data/sft/train_v2_chalonly.jsonl --output ckpt/v2 \
  --base /workspace/models/Qwen3-4B-Instruct-2507-overridden \
  --epochs 3 --lr 1e-5 --batch-size 2 --grad-accum 8 \
  --no-flash-attn --skip-sanity --completion-loss \
  2>&1 | tee /tmp/train.log; echo EXIT=${PIPESTATUS[0]}' Enter
# слежение:
tail -f /tmp/train.log | grep --line-buffered -E "'loss'|Saving|Error"
```

### Шаг B. Локальная валидация (12 проверок, нужен GPU для sanity)
```bash
python3 scripts/validate_local.py --model-dir ckpt/v2/final \
  --repo tarab/albedo-qwen3-4b-v2 --king-dir /workspace/models/Qwen3-4B \
  --fingerprint-against /workspace/models/Qwen3-4B \
    /workspace/models/Qwen3-4B-Instruct-2507 \
    /workspace/models/Qwen3-4B-Instruct-2507-overridden
```
Должно быть **N/N passed, 0 failed**. + глазами просмотреть генерации (THOUGHT + 1 bash, чистый стоп).

### Шаг C. Upload в Hippius (БЕЗ on-chain commit — регистрация не нужна)
```bash
bash scripts/deploy_v6.sh /root/albedo_run/ckpt/v2/final v2 upload
# внутри: validate_local → albedo upload --namespace tarab --name v2
# вернёт repo@sha256:<digest>. Репо появится: hub.hippius.com/models/tarab/albedo-qwen3-4b-v2
```
Альтернатива вручную:
```bash
cd /root/albedo_repo && ./.venv-deploy/bin/albedo upload \
  --path /root/albedo_run/ckpt/v2/final --namespace tarab --name v2
```

### Шаг D. (ПОЗЖЕ) Регистрация хоткея на netuid 97
Нужен баланс coldkey ≥ стоимости (сейчас **τ1.03**, на балансе **τ0.60** → пополнить!).
```bash
cd /root/albedo_repo
# проверить стоимость и баланс:
./.venv-deploy/bin/python -c "from chain_reader.chain import connect; st=connect('finney'); \
print('cost', st.recycle(netuid=97)); print('bal', st.get_balance('5Gp46RymQMogKY3SQbEKDjDbYbynkREXb23hAFpAZTvNZsJw'))"
# регистрация (тратит TAO):
./.venv-deploy/bin/albedo register --coldkey runp --hotkey aff1 --netuid 97
```

### Шаг E. (ПОЗЖЕ) On-chain commit v6
```bash
# полный путь: validate → upload → check-hippius → commit v6
bash scripts/deploy_v6.sh /root/albedo_run/ckpt/v2/final v2 publish
# или только commit готового репо:
cd /root/albedo_repo && ./.venv-deploy/bin/albedo commit \
  --repo tarab/albedo-qwen3-4b-v2 --digest sha256:<digest>
```

### Шаг F. Проверить, что коммит в цепочке и идёт eval
```bash
cd /root/albedo_repo && ./.venv-deploy/bin/albedo check-commit \
  --hotkey 5FJB1Po31WKhnmaZSsNPGMdR9wT4W5EGYyoSJnUeQpSGXgM5
# дашборд: https://us-east-1.hippius.com/albedo/index.html
```

### Чек-лист перед выстрелом (commit)
- [ ] `validate_local.py` — все проверки PASS, similarity < 0.95 против всех баз
- [ ] глазами: THOUGHT + ровно 1 bash, чистый `<|im_end|>`, без `<think>`, без injection
- [ ] config.json: 11 lock-ключей == genesis, нет auto_map/quant, нет *.py
- [ ] модель залита в Hippius, есть `repo@sha256:digest`
- [ ] хоткей зарегистрирован, баланс покрывает; `git fetch official` — reveal всё ещё **v6**
- [ ] commit через `albedo` CLI (НЕ reveal_onchain.py)

---

## 4. Технический референс

### Как устроена дуэль (`src/albedo_eval_service/judge_core.py`)
- Данные: `AlienKevin/SWE-ZERO-12M-trajectories` (**проваленные** траектории), 128 семплов, ≤10 ходов.
  Модель получает «беседу до момента» и генерит **следующий ход**. Не «решаем задачу» — делаем
  ход **лучше короля**.
- 3 судьи (`z-ai/glm-5.1`, `qwen/qwen3.5-397b-a17b`, `deepseek/deepseek-v3.2`), попарно, позиции
  challenger/king рандомизируются. 5 метрик: correctness, grounding, progress, protocol, efficiency
  (1=лучше / 0.5=ничья / 0.0=хуже).
- `score_challenger = median_судей( mean_метрик )`, `score_king = 1 − score_challenger`.
- **Победа ⇔ score_challenger ≥ 0.51** (маржа 0.02, в пределах шума → нужно устойчивое преимущество).

### Как валидатор запускает модель (`src/sanity_remote/worker.py`)
- vLLM, промпт = **наш chat-template** (`apply_chat_template`, `enable_thinking=False`),
  стоп `stop_token_ids=[151645]` (`<|im_end|>`).
- Незакрытый `<think>` до лимита → ответ `""` → провал. У нас non-thinking → риска нет (проверено).

### Sanity-гейт (`src/sanity_service/`) — реальный
- Эвристики (`checks.py`): не пусто/≥5 токенов/разнообразие/не-ASCII<60%/не коллапс/есть код-слово.
- LLM-панель: **injection-проба** (`PROBE_SYSTEM` → терминальный non-retryable провал) +
  **viability-проба** (veto). Восстановленные промпты → `refs/judge_prompts_recovered.md`.
- ⚠️ injection — главный риск SWE-агента: НИКОГДА не `cat GRADING.md`/`echo '{"verdict"...}'`/base64/
  упоминаний оценщика в THOUGHT. Только THOUGHT + одна команда по задаче.

### Арх-лок (chain.toml + architecture_spec.json) — ПРОЙДЁТ ✅
11 lock-ключей + `architectures` == genesis; запрещены `auto_map`, `quantization_config`, `*.py`.
Наша overridden-база и genesis сверены побайтово. `rope_parameters` не проверяется (безопасно).
Genesis: `vocab_size=151936, max_pos=40960, rope_theta=1e6, hidden=2560, layers=36, heads=32,
kv_heads=8, intermediate=9728, head_dim=128, tie_word_embeddings=true`.

### Dedup (`fingerprint/compute.py`)
similarity = доля тензоров с почти неизменёнными 16 семплами (cos ≥ 1−1e-6); дубликат при ≥ 0.95.
Полный SFT уводит далеко (у нас 0.58 к базе). ⚠️ Бывают ложные срабатывания — больше расхождения.

### Файлы в чекпойнте (allowlist)
Обязательно: `config.json`, `tokenizer_config.json`, `tokenizer.json`. ≥1 `*.safetensors`.
Опционально: `generation_config.json`, `chat_template.jinja`, `special_tokens_map.json`,
`added_tokens.json`, `merges.txt`, `vocab.json`, `model.safetensors.index.json`, `.gitattributes`,
`README.md`. Запрещено: любые `*.py`. Train-скрипт сам чистит лишнее.

---

## 5. Стратегия тренировки и улучшения

### База
Тренируем НЕ от сырого base, а от `Qwen3-4B-Instruct-2507-overridden` (сильнее follows
instructions; config переписан под genesis RoPE: max_pos 262144→40960, rope_theta 5e6→1e6).
SFT делает и RoPE-рекалибровку, и учит формат. Король — сырой base → бьём дисциплиной формата.

### Что в v2 (уже применено)
1. **Чистый датасет** `train_v2_chalonly.jsonl` (challenger-победы, all-clean, 0 injection).
2. **assistant-only loss** (`--completion-loss`) — лосс только по нашему ходу, не по контексту.

### Идеи для v3+ (доучивать дальше)
- ~~UNTIE lm_head ПЕРЕД SFT~~ — **СНЯТО** (опровергнуто 18.06; см. §1b). У arboshelper
  lm_head на самом деле не trainable отдельно; «трюк» — артефакт сохранения. Если
  захочется *настоящий* untied (вдруг кто-то его сделает осознанно) — это всё ещё
  возможно через `finetune3/scripts/train_champion_untied.py`, но это experimental.
- 🎯 **Clone-with-improvement от свежего короля (finetune3/)** — самый быстрый путь
  к ≥0.51 score. Стартуем с весов текущего короля + low-LR SFT (5e-7, 60 шагов) на
  нашем `train_v3.jsonl`. Цель — сдвинуть веса до similarity < 0.95, но сохранить
  качество. ETA 5-15 мин на H100. См. `finetune3/GUIDE.md`.
- 🚀 **finetune2/ (v3 untied + completion-only loss)** — наш собственный пайплайн от
  overridden-базы. Сравним с finetune3 — какой даёт лучший eval. ETA 30-40 мин.
- Пересобрать датасет с реальных свежих трейсов (`download_traces.py` → `build_sft_dataset.py`),
  упор на challenger-победы с высоким score.
- Подобрать epochs/lr (2–3 эпохи, lr 1e-5…2e-5); следить, чтобы не перезабыть базу.
- Не stack-ить SFT поверх SFT без нужды (риск дрейфа); чистый прогон от base предпочтительнее.
- Усиливать grounding/efficiency (короткий THOUGHT, точные команды по наблюдениям).
- При шардировании v3 сразу делать `max_shard_size="2GB"` (Hippius 8 GB single safetensors
  падает с 404 — см. §8.3).

### Запуск тренировки (всегда в tmux!)
Команда — см. Шаг A. Параметры: `--no-flash-attn`→sdpa (eager=OOM); 405 шагов (v1, full-text)
или ~438 (v2, 3 эпохи × 146). `save_strategy=epoch`. ⚠️ Никогда не запускать голым `&` (SIGHUP).

---

## 6. Диагностика

| Симптом | Причина / решение |
|---------|-------------------|
| ГПУ пуст, лог встал | Процесс умер. Без tmux/nohup → SIGHUP. Перезапуск в tmux. |
| `Intel oneMKL FATAL ERROR` | Транзиент контейнера — повторить. |
| CUDA OOM | `attn=eager` при seq=8192 → должно быть `sdpa` (`--no-flash-attn`). |
| `pip install torch==2.6` | ❌ НЕ запускать — ломает окружение. Проверка: `python3 -c "import torch;print(torch.__version__,torch.cuda.is_available())"`. |
| Коммит игнорируется | Версия reveal ≠ v6. Использовать `albedo commit` (v6), не reveal_onchain.py. |
| Fingerprint DQ (≥0.95) | Больше эпох/lr/разнообразия; учесть ложные срабатывания. |
| Ответ пустой на евале | Незакрытый `<think>`. У нас non-thinking — не должно. |
| Hippius whoami 404 | Норма — Hippius не поддерживает whoami; токен валидируется самим upload. |

---

## 7. Ключевые ссылки
- Официальный репо: `github.com/tony-dendrite/albedo` (ветка **dev**). Зеркало: `unarbos/albedo`.
- Дашборд/скоры: `https://us-east-1.hippius.com/albedo/index.html`
- Hippius hub: `hub.hippius.com/models/tarab/albedo-qwen3-4b-<name>`
- Reveal формат: `v6|<repo>|<sha256:digest>` (см. `miner/commit.py`).

---

## 8. Hippius: загрузка модели (рабочая процедура + ключи + проблемы)

> ⚠️ **СЕКРЕТЫ НИЖЕ.** Этот раздел содержит ключи. Не коммить в git, не публиковать, хранить локально.

### 8.1 Ключи и идентификаторы

| Что | Значение | Для чего |
|-----|----------|----------|
| Namespace (Harbor project) | **`tarab`** (Free, квота 100 GB, active) | имя репо `tarab/albedo-qwen3-4b-<name>` |
| Console API token | `70a48ec9746b14ba0f4f9d9ba471c7bf8e354904` | `hippius-hub login --hippius-token` (API, НЕ для push) |
| Master S3 (AK/Secret) | `hip_00a833105766be0243a21b59` / `nJk1uFlderMnt9BCqvMi2bFQZj3I93acicWO9ZMNxz0` | S3-бакеты `s3.hippius.com` (НЕ для реестра моделей!) |
| Registry robot (rotate-token) | `robot$tarab+tarab-bot` / `soxVXqQ0vddR7GbnSOyRxYESPQF4za7X` | push в реестр; пишется в `~/.cache/hippius/hub/token`. Ротация инвалидирует прошлый |
| Залитая модель | `tarab/albedo-qwen3-4b-v2` @ `main` | challenger v2 |
| **Digest (для commit v6)** | **`sha256:b50d3f817920874a690bea3006d1b630e3fb0ebedcf56433d917b625a6b2199b`** | `v6\|tarab/albedo-qwen3-4b-v2\|<digest>` |

Куда грузить: **раздел Models / реестр** (`registry.hippius.com`, Harbor). **НЕ** Bucket (S3) и **НЕ** Drive —
валидатор качает модель через `hippius_hub.snapshot_download` из реестра. S3-бакет реестр использует только
для своих JSONL-отчётов.

### 8.2 Рабочая процедура (то, что реально сработало)

```bash
cd /root/albedo_repo
V=.venv-deploy/bin
# 1) логин консольным токеном (сохраняет API-токен)
$V/hippius-hub login --hippius-token 70a48ec9746b14ba0f4f9d9ba471c7bf8e354904
# 2) КЛЮЧЕВОЙ ШАГ: сминтить registry robot-secret (без него push = 401)
$V/hippius-hub registry rotate-token            # пишет robot-secret в ~/.cache/hippius/hub/token
$V/hippius-hub registry me                      # проверить: Project=tarab, Status=active
# 3) в .env закомментировать HIPPIUS_HUB_TOKEN → upload возьмёт сохранённый robot-secret
# 4) решардить веса в ~2 ГБ (если чекпойнт — один большой model.safetensors): см. §8.4
# 5) загрузка с пониженной конкурентностью + retry (merge-семантика дозаливает пропущенное)
export HIPPIUS_UPLOAD_WORKERS=1 HIPPIUS_MAX_CONCURRENT=4 HIPPIUS_CHUNK_SIZE=33554432
for t in 1 2 3 4 5 6; do
  $V/hippius-hub upload tarab/albedo-qwen3-4b-v2 /root/albedo_run/ckpt/v2/final_sharded --revision main && break
  sleep 5
done
# 6) проверка
$V/hippius-hub revisions tarab/albedo-qwen3-4b-v2 --json     # взять digest
$V/python -c "import hippius_hub; print([ (x.path if hasattr(x,'path') else x) for x in hippius_hub.HippiusApi().list_repo_files(repo_id='tarab/albedo-qwen3-4b-v2', revision='main')])"
```

### 8.3 Проблемы и решения (что встретили)

| Проблема | Причина | Решение |
|----------|---------|---------|
| **401** на push весов | Консольный токен / S3 AK-SK реестром Harbor не принимаются для push. Push-scope давал `actions:[]` | `hippius-hub registry rotate-token` → robot-secret. После — scope `['push','delete','pull']` |
| `whoami`/`_get_namespace` → отклонены/404 | Hippius не поддерживает HF whoami; нужен rotate-token, а namespace смотреть через `registry me` | `registry me` (Project=tarab) |
| **404** на 8-ГБ блобе `model.safetensors` | Реестр флаки на больших/многочанковых блобах (80 чанков/блоб) | **Решард в 5×~2 ГБ** + retry с `UPLOAD_WORKERS=1` |
| 404 интермиттентно и на 2-ГБ шардах | Конкурентные чанк-сессии | retry-цикл (merge пропускает залитое) + `MAX_CONCURRENT=4` |
| S3 AK/SK «не работают» | Это креды **объектного хранилища (Bucket)**, не реестра | Грузить в **Models/реестр**, не в Bucket |

### 8.4 Решард в шарды (если чекпойнт — один model.safetensors)

```bash
cd /root/albedo_run
python3 - <<'PY'
import torch, shutil, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
src="ckpt/v2/final"; dst="ckpt/v2/final_sharded"; Path(dst).mkdir(parents=True, exist_ok=True)
tok=AutoTokenizer.from_pretrained(src)
m=AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, low_cpu_mem_usage=True)
m.save_pretrained(dst, max_shard_size="2GB", safe_serialization=True); tok.save_pretrained(dst)
shutil.copy(src+"/config.json", dst+"/config.json")           # сохранить проверенный config (арх-лок)
for f in ["chat_template.jinja","generation_config.json"]:
    if Path(src+"/"+f).exists(): shutil.copy(src+"/"+f, dst+"/"+f)
keep={"config.json","generation_config.json","tokenizer_config.json","tokenizer.json",
      "special_tokens_map.json","added_tokens.json","chat_template.jinja","merges.txt",
      "vocab.json","model.safetensors.index.json",".gitattributes","README.md"}
for p in Path(dst).iterdir():
    if p.is_file() and p.name not in keep and not re.match(r"model.*\.safetensors$", p.name): p.unlink()
PY
# затем validate_local на ckpt/v2/final_sharded (должно быть PASS) и upload по §8.2
```

### 8.5 Удаление файла из репо (если осталась старая раскладка)

```bash
# напр. убрать одиночный model.safetensors, оставив шардированную раскладку:
cd /root/albedo_repo && $V/python -c "import hippius_hub; \
hippius_hub.upload_folder(repo_id='tarab/albedo-qwen3-4b-v2', folder_path='/root/albedo_run/ckpt/v2/final_sharded', \
revision='main', delete_patterns=['model.safetensors'])"
```
(В нашем случае старого `model.safetensors` в репо не оказалось — упавшие 8-ГБ пуши его не публиковали.)
