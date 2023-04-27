#!/bin/bash

export PYTHONPATH=.
python test_evaluation_purity.py -e nmi
python test_evaluation_purity.py -e ari
python test_evaluation_purity.py -e purity
echo 'Done...'
