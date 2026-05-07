UV := uv
RUN := $(UV) run

# Основной поток продукта — `make app` (Streamlit Wizard).
# Таргеты prepare / features / train / predict / eval оставлены для CLI-режима
# и регрессионных прогонов на эталонной выборке `data/source/`.

.PHONY: install install-m2 test test-fast prepare features-core features-full train-m1 train-m2 predict-m1 predict-m2 eval-m1 eval-m2 app clean help

help:
	@echo "Основная команда: make app — запуск Streamlit-приложения"
	@echo "См. README.md для подробностей и команд без make"

install:
	$(UV) sync

install-m2:
	$(UV) sync --extra m2

test:
	$(RUN) pytest tests/ -v

test-fast:
	$(RUN) pytest tests/ -v -m "not slow and not integration"

prepare:
	$(RUN) python -m src.prepare.pipeline

features-core:
	$(RUN) python -m src.features.extractor --stage=core

features-full:
	$(RUN) python -m src.features.extractor --stage=full

train-m1:
	$(RUN) python -m src.pipeline.train --features=core --out=models/m1/

train-m2:
	$(RUN) python -m src.pipeline.train --features=full --out=models/m2/

predict-m1:
	$(RUN) python -m src.pipeline.predict --models=models/m1/ --out=data/predictions_m1.json

predict-m2:
	$(RUN) python -m src.pipeline.predict --features=full --models=models/m2/ --out=data/predictions_m2.json

eval-m1:
	$(RUN) python -m src.pipeline.evaluate --predictions=data/predictions_m1.json --out=reports/metrics_m1.json

eval-m2:
	$(RUN) python -m src.pipeline.evaluate --predictions=data/predictions_m2.json --out=reports/metrics_m2.json

app:
	$(RUN) streamlit run src/app/streamlit_app.py

clean:
	rm -rf data/originals data/stripped data/features data/splits
	rm -f data/ground_truth.json data/predictions*.json
	rm -rf models reports .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Папка data/source/ с исходниками НЕ удалена — это входные данные"
