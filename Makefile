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