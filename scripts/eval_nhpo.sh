#!/bin/bash

for i in {1..10}
do
	for n in {modular,nmn}
	do
		HPODIR="hyperopt/nhpo/run-$i"
		RESDIR="$HPODIR/$n-results"
		python scripts/vqaCustomEval.py \
			val2014 \
			"$RESDIR/$n-results.json" \
			--result-dir $RESDIR
	done
done