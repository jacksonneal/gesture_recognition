# CS6140-Final-Project

Fall 2021

Code authors
-----------
Jackson Neal  
Arshad Khan  
Zachery Hindley

Installation
------------
These components are installed:

- Python 3.7.9
- Pip 20.1.1

Environment
-----------
Example ~/.bash_aliases:

```
alias python=python3
alias pip=pip3
```

Execution
---------
All of the build & execution commands are organized in the Makefile.

1) Unzip project file.
2) Open command prompt.
3) Navigate to directory where project files unzipped.
4) Create and activate a virtual environment.
5) Install dependencies:
    - `make install`
    - installs dependencies from `requirements.txt` using pip
6) Compile with Cython:
    - `make cython`
7) Join separate gesture datasets (only run once):
    - `make join-ds src=.\datasets\initial dest=.\datasets\joined\joined.csv`
8) Generate train and test datasets (only run once):
    - `make train-test-split src=.\datasets\joined\joined.csv dest=.\datasets\train_test_split test_pct=0.20`
9) Run tests through pytest:
    - `make test`
10) View all run options and parameters:
    - `python -m models -h`
11) Run DecisionTree Train and Save:
    - `python -m models decision-tree train .\datasets\sample_problem\train.csv .\serialized\dt\tree.json`
12) Run DecisionTree Load and Test:
    - `python -m models decision-tree test .\serialized\dt\tree.json .\datasets\sample\test.csv`
13) Run Naive Bayes Classifier Train and Save:
    - `python -m models bayes train .\datasets\sample_problem\train.csv .\serialized\bayes\bayes.json`
14) Run Naive Bayes Classifier Load and Test:
    - `python -m models bayes test .\serialized\bayes\bayes.json .\datasets\sample_problem\test.csv`
15) Run Ensemble Train and Save:
    - `python -m models ensemble train .\datasets\sample_problem\train.csv .\serialized\ensemble\`
16) Run Ensemble Load and Test:
    - `python -m models ensemble test .\serialized\ensemble\ .\datasets\sample_problem\test.csv`
