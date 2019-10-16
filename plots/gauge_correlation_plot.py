import argparse
import json
import matplotlib.pyplot as plt


def get_args():
	descr = """Plot graph comparing Gauge module's accuracy and final NMN accuracy"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('input_log')
	return parser.parse_args()

def norm_factors(values):
	min_val = min(values)
	max_val = max(values)
	return min_val, 1/(max_val-min_val+1e-10)

normalize = lambda vs, d, s: [ (v-d)*s for v in vs ]

def main(args):

	with open(args.input_log) as fd:
		data = json.load(fd)

	plt.figure()
	plt.title('Influence of Find module')
	plt.xlabel('Find stage (epoch)')
	plt.ylabel('Relative Accuracy')

	epochs = [ e+1 for e in data['epoch'] ]
	mean   = data['nmn_top_1_mean']
	std    = data['nmn_top_1_std']

	displ, scale = norm_factors(mean)
	mean   = normalize(mean, displ, scale)
	std    = normalize(std, 0, scale)

	y1 = [ m-2*d for m, d in zip(mean, std) ]
	y2 = [ m+2*d for m, d in zip(mean, std) ]
	plt.fill_between(epochs, y1, y2, alpha=0.3)
	plt.plot(epochs, mean, label='End2end NMN $\pm 2 \sigma$')

	top_1 = data['find_top_1']
	top_1 = normalize(top_1, *norm_factors(top_1))
	plt.plot(epochs, top_1, label='Gauge validation')

	plt.yticks([], [])
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main(get_args())
