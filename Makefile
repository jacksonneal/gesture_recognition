# Makefile for Gesture Recognition project.

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
	python setup.py build_ext --inplace &&

ensemble-train-test: ensemble-train ensemble-test

ensemble-train:
	 python -m models ensemble train .\datasets\train_test_split\train.csv .\serialized\ensemble\ \
 		--k 9342 --n-dt 100 --n-nb 100 --boost --nf 40 --max-split-eval 10 \
 		--max-depth 25 --min-split 10 --gini

ensemble-test:
	python -m models ensemble test \
 		.\serialized\ensemble\ .\datasets\train_test_split\test.csv --save .\output\ensemble\

