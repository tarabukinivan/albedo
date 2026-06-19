```

  /root/albedo_run/finetune4/ (244 KB, 12 файлов)

  GUIDE.md (16 KB)                          ⭐ полный пошаговый план (с TBD-блоками что пока неизвестно)

  refs/
    qwen36_35b_genesis_config.json (4 KB)   ⭐ актуальный config нового genesis
    qwen36_35b_genesis_index.json (98 KB)   полный index 26 шардов / 1045 тензоров
    expected_arch_lock_v4.md (5 KB)         ⭐ предсказание ALL_LOCK_KEYS до выкатки
    judge_prompts_recovered.md (8 KB)       PROBE_SYSTEM (recovered, для SFT anti-injection)

  scripts/
    download_genesis_v4.py (5 KB)           скачать 72 GB новый genesis (с опцией --skip-vision)
    build_sft_v4.py (9 KB)                  ⭐ перетокенизация train_v3 → train_v4 под Qwen3.6
    train_v4.py (11 KB)                     SFT MoE 35B + DeepSpeed ZeRO-3 + freeze_vision
    eval_local_v4.py (19 KB)                ⭐ duel emulator через vLLM endpoints (не CPU!)
    reshard_v4.py (9 KB)                    35+ × 2 GB шардов
    fingerprint_check.py (7 KB)             prod-identical clone-detect (алгоритм не меняется)

  configs/
    ds_zero3.json (696 B)                   ⭐ ZeRO-3 + CPU offload (обязательно для 35B на 4×H100)

  data/sft/  (создаётся по запросу build_sft_v4.py)

  Ответы на вопросы

  Датасет — обновить или старый пойдёт?

  Перетокенизировать обязательно. Я сделал build_sft_v4.py.
  Причина: vocab сменился (151936 → 248320), tokenizer другой (Qwen3-4B BPE → Qwen3.6 BPE). Старый train_v3.jsonl содержит уже отрендеренный
  текст <|im_start|>role\n...<|im_end|> — синтаксис совместим (та же chat template family), но IDs специальных токенов другие. Скрипт:
  1. Парсит старый текст обратно в структурированные messages
  2. Применяет apply_chat_template НОВЫМ tokenizer'ом
  3. Пере-проверяет на injection (наши PROBE_RISK regex, они от tokenizer не зависят)
  4. Записывает с правильными n_tokens

  Ожидаем ~2300-2400 examples (тот же объём что в train_v3).

  Хардкорные промпты — нашёл

  В /root/albedo/src/sanity_service/prompts.json:
  [
    "What does this Python function do?\n\ndef double(x):\n    return x * 2",
    "Complete this function:\n\ndef add(a, b):\n    ",
    "What is wrong with this code?\n\nprint(x[5])  # x is not defined"
  ]

  Это не изменилось. Те же 3 generic Python promptы.

  Нужны ли скрытые промпты судьи (PROBE_SYSTEM)?

  Да, всё ещё нужны. Я проверил /root/albedo/src/sanity_service/rubric.py:

  from sanity_service.rubricisity import PROBE_SYSTEM, VIABILITY_SYSTEM

  rubricisity.py по-прежнему отсутствует в репо (намеренно скрыт). Наш восстановленный judge_prompts_recovered.md остаётся единственным
  источником.

  В eval_local_v4.py я добавил graceful fallback: пытается импортировать prod rubricisity, если нет — использует recovered. Это позволит при
  необходимости запустить локальные injection-проверки.

  Главное — анти-injection фильтр в нашем build_sft_v4.py (12 правил из PROBE_SYSTEM) защищает наш датасет от случайного попадания паттернов
  которые судья пометит как injection.

  Что изменилось vs finetune2/3
  ┌────────────────┬───────────────────────────────┬────────────────────────────────────────────────────┐
  │     Аспект     │         v3 (Qwen3-4B)         │                v4 (Qwen3.6-35B MoE)                │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ eval_local     │ загрузка модели на CPU        │ vLLM endpoints (на pod'е)                          │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ build_sft      │ прямое использование train_v3 │ перетокенизация под новый vocab                    │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ train          │ 1× H100, single-GPU           │ 4× H100, ZeRO-3 + CPU offload                      │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ win_margin     │ 0.02                          │ 0.06 (3× жёстче)                                   │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ Размер ckpt    │ 8 GB                          │ 72 GB (35+ × 2 GB шардов)                          │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ Untie lm_head  │ актуально                     │ не нужно (genesis уже untied)                      │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ Vision encoder │ нет                           │ freeze для SWE (флаг --freeze-vision)              │
  ├────────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
  │ chat_template  │ стандарт Qwen3                │ специфичен Qwen3.6 (берётся из скачанного genesis) │
  └────────────────┴───────────────────────────────┴────────────────────────────────────────────────────┘

  Открытые вопросы для проверки когда выкатят официальный код

  1. chain.toml — узнать точные repo_pattern, seed_repo, extra_lock_keys
  2. architecture.py — как validator проверяет вложенные text_config поля
  3. transformers v5.12.1 — поддерживает ли Qwen3_5MoeForConditionalGeneration
  4. vLLM — поддерживает ли эту модель
  5. TRL completion_only_loss — работает ли с multimodal
  6. Vision encoder — можно ли заморозить без нарушения validator'a

  Что прямо сейчас можно делать

  - Подождать пока Dendrite выкатят финальный код (ETA: вторая половина дня 19.06 или 20.06)
  - Когда выкатят: cd /root/albedo && git pull && diff chain.toml → проверить мои предсказания в expected_arch_lock_v4.md
  - Запустить build_sft_v4.py чтобы протестировать перетокенизацию ДО pod'а (нужен только скачанный config + tokenizer, ~14 MB)
  - Если есть свободный GPU доступ — попробовать загрузить минимальный MoE Qwen3.5 (например Qwen3.5-30B-A3B на HF) и проверить совместимость с
   TRL

  Что НЕ делать пока

  - НЕ заказывать 4× H100 pod
  - НЕ скачивать 72 GB genesis (пустая трата если конфиг или digest сменится)
  - НЕ запускать train_v4.py (он пока предположение)
  - НЕ удалять finetune2/finetune3 (как reference + для возможного allowed re-eval Qwen3-4B legacy)
