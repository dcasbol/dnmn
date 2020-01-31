import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

def get_args():
	descr = """Take time data from JSON and show it in a barplot"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('input_log')
	parser.add_argument('--raw-times', action='store_true')
	return parser.parse_args()


NAMES = ['encoder', 'find', 'cache', 'measure', 'describe']

def main(args):

	with open(args.input_log) as fd:
		data = json.load(fd)

	suf = '(raw)' if args.raw_times else '(forward-backward)'
	plt.figure()
	sns.set_palette(sns.color_palette("Dark2"))
	plt.title('Training times '+suf)
	plt.ylabel('Hours')

	bottom = 0
	width = 0.5
	plots = list()

	time_key = 'raw_time' if args.raw_times else 'time'

	for name in NAMES:
		t = data[time_key][name]/3600
		p = plt.bar(0, t, width, bottom=bottom)
		plots.append(p)
		bottom += t

	t = data[time_key]['nmn']/3600
	p = plt.bar(1, t, width)
	plots.append(p)

	plt.xticks((0, 1), ('modular', 'end2end'))
	plt.legend(plots, NAMES+['nmn'])

	plt.show()

if __name__ == '__main__':
	main(get_args())
