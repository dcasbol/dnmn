source envnmntorch/bin/activate

if [ ! -d "hyperopt/" ]
then
	mkdir "hyperopt"
fi

# All modules explore same hyperparameters
python scripts/generate_hpo_candidates.py \
	--selection "nmn" --output "hyperopt/hpo_candidates.json"

for m in {find,encoder}
do
	python optimize_hypers.py $m \
		--candidates "hyperopt/hpo_candidates.json"
done

# Generate cache before optimizing root modules
for ds in {train2014,val2014}
do
	python generate_cache.py \
		--find-module "hyperopt/find/find-hpo-best.pt" \
		--dataset $ds --overwrite
done

for m in {measure,describe}
do
	python optimize_hypers.py $m --candidates "hyperopt/hpo_candidates.json"
done

# Generate result JSON file
python generate_results.py val2014 \
	--encoder "hyperopt/encoder/encoder-hpo-best.pt" \
	--find "hyperopt/find/find-hpo-best.pt" \
	--describe "hyperopt/describe/describe-hpo-best.pt" \
	--measure "hyperopt/measure/measure-hpo-best.pt" \
	--output "hyperopt/direct-modular-results.json"

deactivate

source envpy2/bin/activate

if [ ! -d "hyperopt/direct-modular-evaluation" ]
then
	mkdir "hyperopt/direct-modular-evaluation"
else
	rm -r "hyperopt/direct-modular-evaluation"
	mkdir "hyperopt/direct-modular-evaluation"
fi

python scripts/vqaCustomEval.py \
	val2014 \
	"hyperopt/direct-modular-results.json" \
	--data-dir "data/vqa" \
	--result-dir "hyperopt/direct-modular-evaluation"

deactivate