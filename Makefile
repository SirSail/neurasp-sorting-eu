# Makefile for Order List 5 Numbers Experiments

# Configuration variables
# Datasets sizes to generate and train on. Note: 1000 becomes '1k', 10000 becomes '10k' in folder names.
SIZES ?= 10000

# NeurASP Hyperparameters
ALPHAS ?= 0.1 0.25 0.5 0.75 0.9
EPOCHS_NEURASP ?= 50
LR_NEURASP ?= 0.001

# Neural Baseline Hyperparameters
EPOCHS_NEURAL ?= 50
LR_NEURAL ?= 0.001

# Python interpreter command
PYTHON := uv run python

# Paths
DATA_DIR := data
SCRIPTS_DIR := scripts
RESULTS_DIR := results/experiments

.PHONY: all data train_all train_neural train_neurasp plot clean

# Default target: Run everything
all: data train_all plot

# 1. Generate Data
data:
	@echo "Generating datasets: $(SIZES)..."
	$(PYTHON) $(SCRIPTS_DIR)/data_gen.py --sizes $(SIZES) --out_dir $(DATA_DIR)

# 2. Train Models
train_all: train_neural train_neurasp

train_neural:
	@echo "--- Training Neural Baseline ---"
	@for size in $(SIZES); do \
		if [ "$$size" -ge 1000 ]; then \
			s_str=$$(($$size / 1000))k; \
		else \
			s_str=$$size; \
		fi; \
		echo "Training Neural Baseline on size $$s_str (Epochs: $(EPOCHS_NEURAL), LR: $(LR_NEURAL))..."; \
		$(PYTHON) $(SCRIPTS_DIR)/train.py \
			--data_dir $(DATA_DIR)/$$s_str \
			--epochs $(EPOCHS_NEURAL) \
			--lr $(LR_NEURAL); \
	done

train_neurasp:
	@echo "--- Training NeurASP ---"
	@for size in $(SIZES); do \
		if [ "$$size" -ge 1000 ]; then \
			s_str=$$(($$size / 1000))k; \
		else \
			s_str=$$size; \
		fi; \
		for alpha in $(ALPHAS); do \
			echo "Training NeurASP on size $$s_str (Alpha: $$alpha, Epochs: $(EPOCHS_NEURASP), LR: $(LR_NEURASP))..."; \
			$(PYTHON) $(SCRIPTS_DIR)/train_neurasp.py \
				--data_dir $(DATA_DIR)/$$s_str \
				--epochs $(EPOCHS_NEURASP) \
				--alpha $$alpha \
				--lr $(LR_NEURASP); \
		done; \
	done

# 3. Plot Results
plot:
	@echo "Generating plots and analysis..."
	$(PYTHON) $(SCRIPTS_DIR)/plot_results.py
	$(PYTHON) $(SCRIPTS_DIR)/analyze_results.py

# Clean results (Optional)
clean:
	rm -rf $(RESULTS_DIR)
