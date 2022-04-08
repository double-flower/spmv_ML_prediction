# A script for batch testing SpMV performance

## prerequisites
- [ssget](https://github.com/ginkgo-project/ssget)
- python2.7


## Code overview
- **all_mtx**: the name of all matrices from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- **todo_mtx**: the name of the matrices to be tested
- **run_matrix.py**: scritp for bathc testing

## Usage
- modify file `todo_mtx`
- run command: `python2 run_matrix.py`
