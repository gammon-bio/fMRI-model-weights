# All targets use the project-local venv. Thread caps come from configs/default.yaml (compute.n_threads).
PY      ?= .venv/bin/python
CONFIG  ?= configs/default.yaml
JOBS    ?= 1
STAMP   := $(shell date +%Y%m%d_%H%M%S)

.PHONY: env test reproduce runs aggregate gate archive clean finetune

env:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -e .

test:
	OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $(PY) -m pytest -q

# Archive previous results (never overwrite in place), rerun everything from seeds, aggregate, gate.
reproduce: archive runs aggregate gate

archive:
	@if [ -d results/runs ]; then mkdir -p results/archive/$(STAMP) && \
	  mv results/runs results/archive/$(STAMP)/ && \
	  cp -r results/figures results/*.json results/*.md results/archive/$(STAMP)/ 2>/dev/null; \
	  echo "archived previous results to results/archive/$(STAMP)"; fi

runs:
	$(PY) scripts/run_all.py --config $(CONFIG) --jobs $(JOBS)

aggregate:
	$(PY) scripts/aggregate.py --config $(CONFIG)

gate:
	$(PY) scripts/gate.py --config $(CONFIG)

# Phase 2 repair target: fine-tune a diagnosis head on the reproduce checkpoints (run after `make reproduce`)
finetune:
	$(PY) scripts/finetune.py --jobs $(JOBS)

clean:
	@echo "Not deleting anything automatically. To remove generated outputs run: rm -rf outputs/"
