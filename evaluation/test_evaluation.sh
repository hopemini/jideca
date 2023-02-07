#!/bin/bash

export PYTHONPATH=.
python test_evaluation.py -e nmi
python test_evaluation.py -e ari
echo 'Done...'
