import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def get_args():
	parser = argparse.ArgumentParser(description='Show gauge-nmn correlation plot')
	parser.add_argument('--data-json', default='gauge_corr_data.json')
	return parser.parse_args()

def main(args):

	with open(args.data_json) as fd:
		d = json.load(fd)

	gauge_accs, nmn_accs = zip(*[ (g,n) for g,n in zip(d['gauge_accs'], d['nmn_accs']) if n>0 ])

	r, p = pearsonr(gauge_accs, nmn_accs)
	print('coeff: ', r)
	print('p-value: ', p)

	plt.figure()
	plt.scatter(gauge_accs, nmn_accs)
	plt.show()

if __name__ == '__main__':
	main(get_args())