source envnmntorch/bin/activate

if [ ! -d "hyperopt/modular" ]
then
	mkdir -p "hyperopt/modular"
fi

for m in {find,encoder,measure,describe}
do
	if [ ! -d "hyperopt/modular/$m" ]
	then
		mkdir "hyperopt/modular/$m"
	else
		rm -r "hyperopt/modular/$m"
		mkdir "hyperopt/modular/$m"
	fi

	python scripts/generate_hpo_candidates.py \
		--selection "$m" \
		--output "hyperopt/modular/hpo_candidates-$m.json"
done

# 1st Encoder --> cache
python optimize_hypers.py "encoder" \
	--modular \
	--candidates "hyperopt/modular/hpo_candidates-encoder.json"
for ds in {train2014,val2014}
do
	python generate_cache.py \
		--modular \
		--qenc-module "hyperopt/modular/encoder/encoder-hpo-best.pt" \
		--qenc-config "hyperopt/modular/encoder/encoder-res.dat" \
		--dataset $ds --overwrite
done

# 2nd Find --> cache
python optimize_hypers.py "find" \
	--modular \
	--candidates "hyperopt/modular/hpo_candidates-find.json"
for ds in {train2014,val2014}
do
	python generate_cache.py \
		--modular \
		--find-module "hyperopt/modular/find/find-hpo-best.pt" \
		--find-config "hyperopt/modular/find/find-res.dat" \
		--dataset $ds --overwrite
done

# Now optimize root modules
for m in {measure,describe}
do
	python optimize_hypers.py $m \
		--modular \
		--candidates "hyperopt/modular/hpo_candidates-$m.json"
done

# Generate result JSON file
python generate_results.py val2014 \
	--modular-hpo-dir "hyperopt/modular" \
	--output "hyperopt/adjusted-modular-results.json"

deactivate

source envpy2/bin/activate

if [ ! -d "hyperopt/adjusted-modular-evaluation" ]
then
	mkdir "hyperopt/adjusted-modular-evaluation"
else
	rm -r "hyperopt/adjusted-modular-evaluation"
	mkdir "hyperopt/adjusted-modular-evaluation"
fi

python scripts/vqaCustomEval.py \
	val2014 \
	"hyperopt/adjusted-modular-results.json" \
	--data-dir "data/vqa" \
	--result-dir "hyperopt/adjusted-modular-evaluation"

deactivate