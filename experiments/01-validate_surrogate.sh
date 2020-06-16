source envnmntorch/bin/activate

python train_random_find_modules.py 0 100
python scripts/create_corr_json.py --silent --force-save

N = $(python scripts/get_result_count.py "gauge_corr_data.json")
i=1
while [ "$i" -le "$N" ]; do
  python train_corr.py --corr-data "gauge_corr_data.json"
  i=$(($i + 1))
done

python show_corr_data.py --data-json "gauge_corr_data.json" --criterion "loss"

deactivate
