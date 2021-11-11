# Makefile for Gesture Recognition project.

# Install dependencies
install:
	pip install -r requirements.txt

# Freeze dependencies
freeze:
	pip freeze > requirements.txt

# List dependencies
list:
	pip list

# Run tests suite
test:
	python -m pytest tests

# Join the datasets in the given src directory and place them in the given dest file.
join-ds:
	python -m preprocessing.Preprocessor join $(src) $(dest)

# Split the dataset src file given into the given percentage of test examples and train examples,
# putting the results into the given destination dir.
train-test-split:
	python -m preprocessing.Preprocessor train_test_split $(src) $(dest) $(test_pct)