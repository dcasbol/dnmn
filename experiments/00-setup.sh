# Setup python2 environment
virtualenv -p python2 envpy2
source envpy2/bin/activate
pip install requirements_py2.txt
deactivate

# Setup python3 environment
virtualenv -p python3 envnmntorch
source envnmntorch/bin/activate
pip install requirements.txt

python scripts/img_to_feat.py
python scripts/compute_normalizers.py
deactivate
