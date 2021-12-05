# Makefile for Gesture Recognition project.

# Customize these paths for your environment
# -------------------------------------------
train=.\datasets\train_test_split\train.csv
test=.\datasets\train_test_split\test.csv
# -------------------------------------------

# Install dependencies.
install:
	pip install -r requirements.txt

# Freeze dependencies.
freeze:
	pip freeze > requirements.txt

# List dependencies.
list:
	pip list

# Run tests suite.
test:
	python -m pytest tests

# Join the datasets in the given src directory and place them in the given dest file.
join-ds:
	python -m preprocessing.preprocessor join $(src) $(dest)

# Split the dataset src file given into the given percentage of test examples and train examples,
# putting the results into the given destination dir.
train-test-split:
	python -m preprocessing.preprocessor train_test_split $(src) $(dest) $(test_pct)

# Lint python modules.
lint:
	pylint preprocessing.preprocessor --disable=missing-docstring --disable=no-member

# Run Model
model:
	python -m models $(ARGS)

# Compile with cython
cython:
	python setup.py build_ext --inplace

help:
	python -m models -h

bayes-train-test: bayes-train bayes-test

bayes-train:
	python -m models bayes train $(train) $(test) .\serialized\bayes\bayes.json

bayes-test:
	python -m models bayes test $(train) $(test) .\serialized\bayes\bayes.json \
 		--save .\output\bayes\ --cv

dt-train-test: dt-train dt-test

dt-train:
	python -m models decision-tree train $(train) $(test) .\serialized\dt\dt.json \
		--min-split 1 --max-depth 100000 --max-split-eval 1000 --gini

dt-test:
	python -m models decision-tree test $(train) $(test) .\serialized\dt\dt.json \
 		--save .\output\dt\ --cv

ensembleb-train-test: ensembleb-train ensembleb-test

ensembleb-train:
	 python -m models ensemble train $(train) $(test) .\serialized\ensembleb\ \
 		--k 9342 --n-dt 0 --n-nb 1000 --nf 6 --boost

ensembleb-test:
	python -m models ensemble  test $(train) $(test) .\serialized\ensembleb\ \
 		--save .\output\ensembleb\ --cv

rf-train-test: rf-train rf-test

rf-train:
	python -m models ensemble train $(train) $(test) .\serialized\ensemblerf\ \
		--min-split 2 --max-depth 100000 --gini \
 		--k 9342 --n-dt 2 --n-nb 0 --nf 6 --boost

rf-test:
	python -m models ensemble  test $(train) $(test) .\serialized\ensemblerf\ \
 		--save .\output\ensemblebrf\ --cv

ensemble-mix-train-test: ensemble-mix-train ensemble-mix-test

ensemble-mix-train:
	python -m models ensemble train $(train) $(test) .\serialized\ensemblemix\ \
		--min-split 2 --max-depth 100000 --gini \
 		--k 9342 --n-dt 100 --n-nb 100 --nf 6 --boost

ensemble-mix-test:
	python -m models ensemble  test $(train) $(test) .\serialized\ensemblemix\ \
 		--save .\output\ensemblemix\ --cv

sklearn-svc:
	python -m models.sklearn_svc .\datasets\train_test_split\train.csv .\datasets\train_test_split\test.csv

sklearn-rf:
	python -m models.sklearn_rf .\datasets\train_test_split\train.csv .\datasets\train_test_split\test.csv

