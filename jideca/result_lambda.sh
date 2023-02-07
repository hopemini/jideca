#!/bin/bash
## JIDECA

ITERATION=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )

#python result.py -t real -js
for i in ${ITERATION[@]}; do
	echo "${i}th jideca clustering..."
	python result_lambda.py -js -sl ../saved/log_b02l01re -i ${i}
	python result_lambda.py -js -sl ../saved/log_b02l03re -i ${i}
	python result_lambda.py -js -sl ../saved/log_b02re -i ${i}
	python result_lambda.py -js -sl ../saved/log_b02l07re -i ${i}
	python result_lambda.py -js -sl ../saved/log_b02l09re -i ${i}
	python result_lambda.py -js -sl ../saved/log_b05l01re_23 -i ${i}
	python result_lambda.py -js -sl ../saved/log_b05l03re_23 -i ${i}
	python result_lambda.py -js -sl ../saved/log_b05re_23 -i ${i}
	python result_lambda.py -js -sl ../saved/log_b05l07re_23 -i ${i}
	python result_lambda.py -js -sl ../saved/log_b05l09re_23 -i ${i}
done
