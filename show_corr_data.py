import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

def get_args():
	parser = argparse.ArgumentParser(description='Show gauge-nmn correlation plot')
	parser.add_argument('--data-json', default='gauge_corr_data.json')
	parser.add_argument('--criterion',
		choices=['accs','rel_accs','loss'], default='loss')
	return parser.parse_args()

def main(args):

	with open(args.data_json) as fd:
		d = json.load(fd)

	gauge_k = 'gauge_' + args.criterion
	nmn_k   = 'nmn_'   + args.criterion

	values = [ (nv, gv) for nv, gv in zip(d[nmn_k], d[gauge_k]) if nv>0 ]
	nmn_values, gauge_values = zip(*values)
	if 'accs' in args.criterion:
		gauge_values = [ v*100 for v in gauge_values ]
	print('Showing {} points'.format(len(gauge_values)))

	r, p = pearsonr(gauge_values, nmn_values)
	print('coeff: ', r)
	print('p-value: ', p)

	k = 'accuracy (%)' if 'accs' in args.criterion else 'loss'
	if 'rel' in args.criterion:
		k = 'relative ' + k

	plt.figure()
	gk = 'Sparring ' + k
	nk = 'NMN ' + k
	df = pd.DataFrame(zip(gauge_values, nmn_values),
		columns=[gk,nk])
	sns.regplot(x=gk, y=nk, data=df)
	plt.show()

if __name__ == '__main__':
	main(get_args())
