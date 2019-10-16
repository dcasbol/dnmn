import argparse
import json
import matplotlib.pyplot as plt


def get_args():
	descr = """Plot graph comparing Gauge module's accuracy and final NMN accuracy"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('input_log')
	return parser.parse_args()

def main(args):

	with open(args.input_log) as fd:
		data = json.load(fd)

	plt.figure()
	plt.title('Influence of Find module')
	plt.xlabel('Find module at epoch')
	plt.ylabel('Accuracy (%)')

	epochs = [ e+1 for e in data['epoch'] ]
	mean   = [ m*100 for m in data['nmn_top_1_mean'] ]
	std    = [ s*100 for s in data['nmn_top_1_std'] ]

	y1 = [ m-2*d for m, d in zip(mean, std) ]
	y2 = [ m+2*d for m, d in zip(mean, std) ]
	plt.fill_between(epochs, y1, y2, alpha=0.3)
	plt.plot(epochs, mean, label='End2end NMN')

	top_1 = [ a*100 for a in data['find_top_1'] ]
	top_1_train = [ a*100 for a in data['find_top_1_train'] ]
	plt.plot(epochs, top_1, label='Gauge validation')
	plt.plot(epochs, top_1_train, label='Gauge training')

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main(get_args())
