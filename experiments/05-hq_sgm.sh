source envnmntorch/bin/activate

cp -r "hyperopt/modular" "hyperopt/hq-modular"

for m in {find,encoder,measure,describe}
do
	rm "hyperopt/hq-modular/$m/*.json"
done

# 1st Encoder --> cache
for ds in {train2014,val2014}
do
	python generate_cache.py \
		--modular \
		--qenc-module "hyperopt/hq-modular/encoder/encoder-hpo-best.pt" \
		--qenc-config "hyperopt/hq-modular/encoder/encoder-res.dat" \
		--dataset $ds --overwrite
done

# 2nd Find --> cache
# hard-coded best config from last search
python train.py 'find' \
	--batch-size 428 \
	--dropout 0.16162816413740008 \
	--learning-rate 0.0012365193781241743 \
	--weight-decay 5.112447261016383e-10 \
	--modular --hq-gauge --validate
mv "gauge-find-new.pt" "hyperopt/hq-modular/find/find-hpo-best.pt"
for ds in {train2014,val2014}
do
	python generate_cache.py \
		--modular \
		--find-module "hyperopt/hq-modular/find/find-hpo-best.pt" \
		--find-config "hyperopt/hq-modular/find/find-res.dat" \
		--dataset $ds --overwrite
done

# Now optimize root modules
for m in {measure,describe}
do
	python optimize_hypers.py $m \
		--modular \
		--candidates "hyperopt/hq-modular/hpo_candidates-$m.json"
done

# Generate result JSON file
python generate_results.py val2014 \
	--modular-hpo-dir "hyperopt/hq-modular" \
	--output "hyperopt/hq-modular-results.json"

deactivate

source envpy2/bin/activate

if [ ! -d "hyperopt/hq-modular-evaluation" ]
then
	mkdir "hyperopt/hq-modular-evaluation"
else
	rm -r "hyperopt/hq-modular-evaluation"
	mkdir "hyperopt/hq-modular-evaluation"
fi

python scripts/vqaCustomEval.py \
	val2014 \
	"hyperopt/hq-modular-results.json" \
	--data-dir "data/vqa" \
	--result-dir "hyperopt/hq-modular-evaluation"

deactivate