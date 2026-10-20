PYTHON ?= python3

.PHONY: all october legacy figures verify test help

help:
	@echo "make october  Verify October tables and render October figures/tables"
	@echo "make legacy   Verify and rebuild the May/August release"
	@echo "make all      Rebuild both experiment waves offline"
	@echo "make verify   Check frozen data and both waves' numerical outputs"
	@echo "make figures  Render both waves' figures from the shipped tables"
	@echo "make test     Run offline release-interface tests"

all: legacy october

october:
	$(PYTHON) analysis/build_october.py
	$(PYTHON) analysis/make_october_figures_tables.py

legacy:
	$(PYTHON) analysis/build_canonical.py
	$(PYTHON) analysis/make_figures.py
	PYTHONPATH=analysis $(PYTHON) analysis/make_exploratory_figure.py

figures:
	$(PYTHON) analysis/make_figures.py
	PYTHONPATH=analysis $(PYTHON) analysis/make_exploratory_figure.py
	$(PYTHON) analysis/make_october_figures_tables.py

verify:
	$(PYTHON) analysis/validate_release.py
	$(PYTHON) analysis/build_canonical.py
	$(PYTHON) analysis/build_october.py --check-only

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m unittest discover -s analysis -p test_october_build.py -v
