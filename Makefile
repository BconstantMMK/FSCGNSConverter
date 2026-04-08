.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/black lint/flake8 lint/pylint
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([/a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .pytest_cache

lint/black: ## reformat source code
	python -m black .

lint/flake8: ## check code style with flake8
	flake8 py tests

lint/pylint: ## check style with pylint
	pylint .

lint: lint/black lint/flake8 lint/pylint ## run all linter stages

test: ## run tests quickly with the default Python
	pytest tests

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/fscgnsconverter.rst
	rm -f docs/modules.rst
  # sphinx-apidoc -o docs/code src/fscgnsconverter
	sphinx-apidoc -o docs/code py/FSCGNSConverter
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: ## builds source and wheel package
	python -m build .

install: ## install the package to the active Python's site-packages
	pip install .
