import json
import argparse
import matplotlib.pyplot as plt

def get_args():
	descr = """Take time data from JSON and show it in a barplot"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('input_log')
	return parser.parse_args()

def main(args):

	with open(args.input_log) as fd:
		data = json.load(fd)

	plt.figure()
	plt.title('Training times')
	plt.ylabel('Hours')

	NAMES = ['find', 'cache', 'measure', 'describe', 'encoder']
	bottom = 0
	width = 0.5
	plots = list()

	for name in NAMES:
		t = data['time'][name]/3600
		p = plt.bar(0, t, width, bottom=bottom)
		plots.append(p)
		bottom += t

	t = data['time']['nmn']/3600
	p = plt.bar(1, t, width)
	plots.append(p)

	plt.xticks((0, 1), ('modular', 'end2end'))
	plt.legend(plots, NAMES+['nmn'])

	plt.show()

if __name__ == '__main__':
	main(get_args())
