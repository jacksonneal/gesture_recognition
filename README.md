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
4) Ensure Makefile variables point to correct train and test csv files. Ensure directories are
   created for desired mode (ex: Bayes -> .\serialized\bayes\ .\output\bayes)
5) Create and activate a virtual environment.
6) Install dependencies:
    - `make install`
7) Compile with Cython (Must do initially and on editing .pyx files):
    - `make cython`
8) Join separate gesture datasets (only run once):
    - `make join-ds src=.\datasets\initial dest=.\datasets\joined\joined.csv`
9) Generate train and test datasets (only run once):
    - `make train-test-split src=.\datasets\joined\joined.csv dest=.\datasets\train_test_split test_pct=0.20`
10) Run Bayes:
    - `make bayes-train-test`
11) Run DecisionTree:
    - `make dt-train-test`
12) Run RandomForest:
    - `make rf-train-test`
13) Run Ensemble (Bayes and DecisionTree):
    - `make ensemble-train-test`
19) View all run options and parameters for custom runs through module directly:
    - `make help`
