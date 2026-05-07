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
data/Фото YYYY г/  →  prepare/  →  features/  →  train  →  predict  →  evaluate  →  app
                       │           │             │         │           │
                       ↓           ↓             ↓         ↓           ↓
                  ground_truth   resnet.npz   models/   predictions  reports/
                  /stripped      color.npz                           metrics.json
                  /splits        light.npz
```

См. полный план: `C:\Users\Никита\.claude\plans\objective-lovely-toast.md`.

## Стек

- **CV/ML**: pretrained ResNet-50 (torchvision), scikit-learn (kNN+RF+LR ансамбль), HDBSCAN, OpenCV, Pillow, pillow-heif
- **M2 признаки**: facenet-pytorch (MTCNN+FaceNet), EasyOCR
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

## Метрики (целевой коридор)

| Метрика | M1 (ядро) | M2 (full) |
|---------|-----------|-----------|
| accuracy @ year | ≥ 55% | ≥ 65% |
| accuracy @ (year, season) | ≥ 35% | ≥ 50% |
| macro-F1 @ (year, season) | ≥ 0.25 | ≥ 0.40 |
| MAE, месяцев | ≤ 9 | ≤ 6 |
