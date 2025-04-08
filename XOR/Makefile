# Makefile for XOR Model Project

# Definicja zmiennych
PYTHON = python
GENERATE_DATASET = generate_dataset.py
EVALUATE_MODEL = evaluate_model.py
TEACH_MODEL = teach_model.py
OUTPUT_DIR = XOR_data
TRAINING_DIR = training
SAMPLES = 1000
EPOCHS = 1000
DATE = $(shell powershell -Command "(Get-Date).ToString('yyyy-MM-dd')")



all: generate train evaluate

# Reguła do wygenerowania danych
generate:
	$(PYTHON) $(GENERATE_DATASET) $(SAMPLES) --split-name=train --output-dir=$(OUTPUT_DIR)
	$(PYTHON) $(GENERATE_DATASET) $(SAMPLES) --split-name=val --output-dir=$(OUTPUT_DIR)

# Reguła do trenowania modelu
train: generate
	$(PYTHON) $(TEACH_MODEL) "$(OUTPUT_DIR)/$(DATE).$(SAMPLES)_samples.train" --val-files-dir "$(OUTPUT_DIR)/$(DATE).$(SAMPLES)_samples.val" --epochs $(EPOCHS) --output-dir=$(TRAINING_DIR)

# Reguła do ewaluacji modelu
evaluate: train
	$(PYTHON) $(EVALUATE_MODEL) "$(OUTPUT_DIR)/$(DATE).$(SAMPLES)_samples.train" --model-path "$(TRAINING_DIR)/$(DATE)"

