import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

def get_args():
	parser = argparse.ArgumentParser(description='Show gauge-nmn correlation plot')
	parser.add_argument('--data-json', default='gauge_corr_data.json')
	return parser.parse_args()

def main(args):

	with open(args.data_json) as fd:
		d = json.load(fd)

	nmn_accs   = [ a for a in d['nmn_accs'] if a>0 ]
	gauge_accs = d['gauge_accs'][:len(nmn_accs)]
	gauge_accs = [ a*100 for a in gauge_accs ]
	print('Showing {} points'.format(len(gauge_accs)))

	r, p = pearsonr(gauge_accs, nmn_accs)
	print('coeff: ', r)
	print('p-value: ', p)

	plt.figure()
	vars = d['gauge_vars'][:len(gauge_accs)]
	mv = max(vars)
	lv = min(vars)
	gk = 'Gauge accuracy (%)'
	nk = 'NMN accuracy (%)'
	df = pd.DataFrame(zip(gauge_accs, nmn_accs),
		columns=[gk,nk])
	sns.regplot(x=gk, y=nk, data=df)
	plt.show()

if __name__ == '__main__':
	main(get_args())
