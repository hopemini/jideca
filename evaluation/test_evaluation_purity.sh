#!/bin/bash

export PYTHONPATH=.
python test_evaluation.py -e nmi
python test_evaluation.py -e ari
python test_evaluation.py -e purity
echo 'Done...'
