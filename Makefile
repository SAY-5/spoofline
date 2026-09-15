SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run

.PHONY: setup lint test demo demo-fast clean

setup:
	$(UV) sync

lint:
	$(RUN) ruff check spoofline tests web/scripts
	$(RUN) ruff format --check spoofline tests web/scripts

test:
	$(RUN) pytest

demo:
	$(RUN) spoofline pipeline --profile full

demo-fast:
	$(RUN) spoofline pipeline --profile tiny

clean:
	find data -mindepth 1 -delete 2>/dev/null || true
	find runs -mindepth 1 -delete 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name '__pycache__' -delete 2>/dev/null || true
