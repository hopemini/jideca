#!/bin/bash

export PYTHONPATH=.
python test_evaluation_purity_iter.py -e nmi
python test_evaluation_purity_iter.py -e ari
python test_evaluation_purity_iter.py -e purity
echo 'Done...'
