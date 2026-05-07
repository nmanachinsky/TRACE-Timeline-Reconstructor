# Contributing — TRACE

Спасибо за интерес к проекту! Это курсовая по ИИ, поэтому ограничения чуть строже,
чем в обычном open-source.

## Принципы

1. **Без облачных vision API.** Никаких Claude/GPT-4o/Gemini Vision в коде —
   нарушение требований курса.
2. **Тесты пишутся первыми (TDD).** Минимум 80% покрытия — порог жёсткий, проверяется
   `pytest --cov-fail-under=80`.
3. **Безопасность по умолчанию.** Никаких секретов в коммитах. Используйте
   `.streamlit/secrets.toml` и переменные окружения.
4. **Маленькие фокусированные коммиты.** Conventional Commits: `feat:`, `fix:`,
   `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`, `style:`.

## Как поднять окружение

```bash
git clone https://github.com/nmanachinsky/TRACE-Timeline-Reconstructor.git
cd TRACE-Timeline-Reconstructor
uv sync                                    # core deps
uv sync --extra m2                         # + insightface + EasyOCR (для M2)
```

Положите свои фотографии в `data/source/Фото YYYY г/`, затем:

```bash
make prepare
make features-core
make train-m1 && make predict-m1 && make eval-m1
make app                                   # Streamlit-демо
```

## Workflow для PR

1. Форк → создайте ветку: `feature/<имя>` или `fix/<имя>`.
2. Тесты проходят локально: `uv run pytest tests/`.
3. Откройте PR → заполните шаблон. CI прогонит тесты автоматически.
4. После ревью — squash-merge в main.

## Стиль кода

- Аннотации типов во всех публичных функциях.
- Иммутабельные структуры по умолчанию (`@dataclass(frozen=True)`).
- Функции ≤25 строк, файлы ≤800 строк.
- Самодокументируемые имена; комментарии объясняют **почему**, не **что**.
- Идиомы PEP 8.

## Архитектурные правила

- Слой `prepare/` — только подготовка данных, без ML.
- Слой `features/` — извлечение признаков, кэш в `.npz`.
- Слой `models/` — обёртки над sklearn / HDBSCAN.
- Слой `pipeline/` — CLI-оркестраторы train/predict/evaluate.
- Слой `app/` — Streamlit UI и inference helpers.

Зависимости направлены вниз: `app` → `pipeline` → `models` → `features` → `prepare`.
