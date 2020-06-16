source envnmntorch/bin/activate

if [ ! -d "hyperopt/" ]
then
	mkdir "hyperopt"
fi

python scripts/generate_hpo_candidates.py \
	--selection "nmn" --output "hyperopt/hpo_candidates.json"

python optimize_hypers.py "nmn" --candidates "hyperopt/hpo_candidates.json"

python generate_results.py val2014 \
	--nmn "hyperopt/nmn/nmn-hpo-best.pt" \
	--output "hyperopt/end-to-end-results.json"

deactivate

source envpy2/bin/activate

if [ ! -d "hyperopt/end-to-end-evaluation" ]
then
	mkdir "hyperopt/end-to-end-evaluation"
else
	rm -r "hyperopt/end-to-end-evaluation"
	mkdir "hyperopt/end-to-end-evaluation"
fi

python scripts/vqaCustomEval.py \
	val2014 \
	"hyperopt/end-to-end-results.json" \
	--data-dir "data/vqa" \
	--result-dir "hyperopt/end-to-end-evaluation"

deactivate
