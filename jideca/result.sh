#!/bin/bash
## JIDECA

ITERATION=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" )

#python result.py -t real -js
for i in ${ITERATION[@]}; do
	echo "${i}th jideca clustering..."
	python result.py -js -sl ../saved/log_b10re -i ${i}
	python result.py -js -sl ../saved/log_b10se -i ${i}
	python result.py -js -sl ../saved/log_b10g02re -i ${i}
	python result.py -js -sl ../saved/log_b10g02se -i ${i}
	python result.py -js -sl ../saved/log_b10g05re -i ${i}
	python result.py -js -sl ../saved/log_b10g05se -i ${i}
	python result.py -js -sl ../saved/log_b10g1re -i ${i}
	python result.py -js -sl ../saved/log_b10g1se -i ${i}
	python result.py -js -sl ../saved/log_b10g10re -i ${i}
	python result.py -js -sl ../saved/log_b10g10se -i ${i}
done
