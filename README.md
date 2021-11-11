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
6) Join separate gesture datasets:
    - `make join-ds src=.\datasets\initial\ dest=.\datasets\joined\joined.csv`
    - only need to run once
7) Generate train and test datasets:
    - `make train-test-split src=.\datasets\joined\joined.csv dest=.\datasets\train_test_split test_pct=0.20`
    - only need to run once
8) Run tests:
    - `make test`
    - runs unit tests through pytest