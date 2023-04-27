#!/bin/bash

export PYTHONPATH=.
python test_evaluation_iter.py -e nmi
python test_evaluation_iter.py -e ari
echo 'Done...'
