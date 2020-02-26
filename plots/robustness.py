import argparse
import re
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MODULE_NAMES = ['find','measure','describe','encoder']

def get_args():
	parser = argparse.ArgumentParser(description='Modular robustness plot')
	parser.add_argument('hpo_dir')
	parser.add_argument('--nmn-hpo')
	return parser.parse_args()

def _get_acc(fn):
	m = re.search('^[a-z]+-(\d{3}\.\d).+_log\.json$', os.path.basename(fn))
	return float(m.group(1))

def _get_accs_dict(path, name=None):
	pat = path if name is None else os.path.join(path, name)
	pat = os.path.join(pat, '*_log.json')
	name = 'nmn' if name is None else name
	return [ { 'name':name, 'accuracy':_get_acc(fn) } for fn in glob.glob(pat) ]

def main(args):

	if args.nmn_hpo is None:
		args.nmn_hpo = os.path.join(args.hpo_dir, 'nmn')

	data = list()
	for name in MODULE_NAMES:
		data.extend(_get_accs_dict(args.hpo_dir, name))
	data.extend(_get_accs_dict(args.nmn_hpo))

	data = pd.DataFrame(data)
	sns.swarmplot(x='name', y='accuracy', data=data)
	plt.show()

if __name__ == '__main__':
	main(get_args())