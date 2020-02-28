#!/bin/bash

NEVALS=5
for i in {1..10}
do
	HPODIR=hyperopt/nhpo/run-$i
	if [ ! -d $HPODIR ]
	then
		mkdir -p $HPODIR
	fi

	# Hyperparameter optimization
	CANDIDATES="${HPODIR}/hpo_candidates.json"
	python scripts/generate_hpo_candidates.py $NEVALS --output $CANDIDATES --seed $i

	for m in {find,encoder,nmn}
	do
		python optimize_hypers.py $m \
			--target-dir $HPODIR \
			--candidates $CANDIDATES
	done

	# Generate cache before optimizing root modules
	for ds in {train2014,val2014}
	do
		python generate_cache.py \
			--find-module "${HPODIR}/find/find-hpo-best.pt" \
			--dataset $ds --overwrite
	done

	for m in {measure,describe}
	do
		python optimize_hypers.py $m --target-dir "${HPODIR}/$m" --candidates $CANDIDATES
	done

	# Generate results JSON files
	python generate_results.py \
		--nmn "${HPODIR}/nmn/nmn-hpo-best.pt" \
		--output "${HPODIR}/nmn-results.json"
	python generate_results.py \
		--encoder "${HPODIR}/encoder/encoder-hpo-best.pt" \
		--find "${HPODIR}/find/find-hpo-best.pt" \
		--describe "${HPODIR}/describe/describe-hpo-best.pt" \
		--measure "${HPODIR}/measure/measure-hpo-best.pt" \
		--output "${HPODIR}/modular-results.json"
done
