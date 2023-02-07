#!/bin/bash
## JIDECA

ITERATION=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )

#python result.py -t real -js
for i in ${ITERATION[@]}; do
	echo "${i}th jideca clustering..."
	python result_mdec.py -sl ../saved/log_mdec_re -i ${i}
	python result_mdec.py -sl ../saved/log_mdec_se -i ${i}
	python result_mdec.py -sl ../saved/log_midec_re -i ${i}
	python result_mdec.py -sl ../saved/log_midec_se -i ${i}
	python result_mdec.py -js -sl ../saved/log_mdeca_re -i ${i}
	python result_mdec.py -js -sl ../saved/log_mdeca_se -i ${i}
done
