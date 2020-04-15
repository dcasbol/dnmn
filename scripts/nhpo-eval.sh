#!/bin/bash

# Run this with python2 environment
for i in {1..10}
do
	HPODIR=hyperopt/nhpo/run-$i
	
	# Evaluate results
	for n in {modular,nmn}
	do
		RESDIR="$HPODIR/$n-results"
		if [ ! -d $RESDIR ]
		then
			mkdir -p $RESDIR
		fi
		mv "$HPODIR/$n-results.json" "$RESDIR/$n-results.json"
		python scripts/vqaCustomEval.py \
			val2014 \
			"$RESDIR/$n-results.json" \
			--result-dir $RESDIR
	done
done