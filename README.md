# TRACE — Timeline Reconstructor

Восстановление хронологии фотографий (год + сезон) без EXIF на основе визуального анализа: ResNet-эмбеддинги, цветовые гистограммы, освещение, лица и OCR. Курсовой проект по ИИ.

## Быстрый старт

```bash
make install                 # зависимости
make prepare                 # стриппинг EXIF + ground truth + train/test split
make features-core           # ResNet + цвет + свет (M1)
make train-m1                # обучить ансамбль kNN+RF+LR
make predict-m1              # предсказание + cluster consensus
make eval-m1                 # accuracy, macro-F1, MAE, confusion matrix
make app                     # Streamlit-демо
```

## Архитектура

```
data/source/Фото YYYY г/  →  prepare/  →  features/      →  train  →  predict  →  evaluate  →  app
                              │           │                 │         │           │
                              ↓           ↓                 ↓         ↓           ↓
                         ground_truth   resnet.npz       models/   predictions  reports/
                         /stripped      color.npz                              metrics.json
                         /splits        light.npz
                                        faces.npz (M2)
                                        ocr.npz (M2)
```

## Стек

- **CV/ML**: pretrained ResNet-50 (torchvision), scikit-learn (kNN+RF+LR ансамбль), HDBSCAN, OpenCV, Pillow, pillow-heif
- **M2 признаки**: insightface (RetinaFace + ArcFace, ONNX), EasyOCR (ru+en)
- **UI**: Streamlit + Plotly
- **Тесты**: pytest, pytest-cov (≥ 80% покрытия)

## Структура

```
src/
├── config.py            # Пути, гиперпараметры, seed
├── prepare/             # Извлечение GT, стриппинг EXIF, train/test split
├── features/            # Извлечение признаков (ResNet, цвет, свет, лица, OCR)
├── models/              # Ансамбль классификаторов + HDBSCAN consensus
├── pipeline/            # train / predict / evaluate
└── app/                 # Streamlit-демо
tests/                   # pytest + ≥ 80% покрытия
data/                    # Вход и артефакты (вне git, см. .gitignore)
models/                  # Артефакты обучения
reports/                 # Метрики и графики
```

## Метрики

| Метрика | Цель M1 | M1 | Цель M2 | M2 | Δ |
|---------|---------|----|---------|----|---|
| accuracy @ year | ≥ 55% | 70.2% | ≥ 65% | **72.6%** | +2.4 пп |
| accuracy @ (year, season) | ≥ 35% | 56.2% | ≥ 50% | **58.3%** | +2.1 пп |
| macro-F1 @ (year, season) | ≥ 0.25 | 0.458 | ≥ 0.40 | **0.473** | +0.015 |
| MAE, месяцев | ≤ 9 | 8.06 | ≤ 6 | **7.74** | −0.32 |

Прогон на 4486 фото (test = 1346, 28 классов). M2-признаки (insightface
face-эмбеддинги: 1196 лиц найдено, EasyOCR: 298 фото с распознанным годом)
улучшают все четыре метрики относительно M1. Цели M2 по accuracy и macro-F1
перевыполнены; MAE приближается к цели, но всё ещё выше из-за тяжёлого хвоста
ошибок на годах, где визуальные сигналы слабые.

Cluster consensus отключён по умолчанию (HDBSCAN на 4486 фото даёт всего ~95
кластеров → большинство test-точек попадают в noise → consensus слегка ухудшает
метрики на 1-2 пп). Включается через `--consensus` в `predict`.

Подробности — `reports/metrics_m{1,2}.json`, confusion matrix —
`reports/metrics_m{1,2}_confusion.png`.
