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
10) Run DecisionTree Train and Save:
    - `python -m models decision-tree train .\datasets\sample_problem\train.csv .\serialized\tree.json`
11) Run DecisionTree Load and Test:
    - `python -m models decision-tree test .\serialized\tree.json .\datasets\sample\test.csv`
12) Run Bagging Ensemble Train and Save:
    - `python -m models bagging train .\datasets\sample_problem\train.csv serialized`
13) Run Bagging Ensemble Load and Test:
    - `python -m models bagging test serialized .\datasets\sample_problem\test.csv`
