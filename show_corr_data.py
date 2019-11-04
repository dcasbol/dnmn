import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

if __name__ == '__main__':

	with open('gauge_corr_data.json') as fd:
		d = json.load(fd)

	gauge_accs, nmn_accs = zip(*[ (g,n) for g,n in zip(d['gauge_accs'], d['nmn_accs']) if n>0 ])

	r, p = pearsonr(gauge_accs, nmn_accs)
	print('coeff: ', r)
	print('p-value: ', p)

	plt.figure()
	plt.scatter(gauge_accs, nmn_accs)
	plt.show()

