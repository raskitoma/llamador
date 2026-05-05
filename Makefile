# llamador — common operations
# Run `make help` for the menu.

SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk -F ':.*?## ' '{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: env
env: ## scaffold .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "wrote .env from template")

.PHONY: dirs
dirs: ## create host data dirs
	@mkdir -p data/models data/config

.PHONY: build
build: env dirs ## build all images (engine takes ~10–25 min)
	$(COMPOSE) build

.PHONY: up
up: env dirs ## start the default stack
	$(COMPOSE) up -d

.PHONY: chat
chat: env dirs ## start with open-webui chat profile
	$(COMPOSE) --profile chat up -d

.PHONY: metrics
metrics: env dirs ## start with prom+grafana
	$(COMPOSE) --profile metrics up -d

.PHONY: down
down: ## stop the stack
	$(COMPOSE) down

.PHONY: logs
logs: ## follow engine logs
	$(COMPOSE) logs -f --tail=200 llama-engine

.PHONY: backend-logs
backend-logs: ## follow backend logs
	$(COMPOSE) logs -f --tail=200 backend

.PHONY: rebuild-engine
rebuild-engine: ## rebuild only the engine image (after pulling TurboQuant updates)
	$(COMPOSE) build --no-cache --pull llama-engine
	$(COMPOSE) up -d llama-engine

.PHONY: pull-default-model
pull-default-model: dirs ## download Qwen3.6-35B-A3B Q4_K_M into ./data/models
	./scripts/pull-model.sh unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf

.PHONY: bench
bench: ## run llama-bench inside the engine container, sweep --n-cpu-moe
	./scripts/bench.sh

.PHONY: nuke
nuke: ## remove containers, networks, named volumes (keeps ./data on host)
	$(COMPOSE) down -v
