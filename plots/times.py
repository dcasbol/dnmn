import os
import argparse
import json
from glob import glob
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

def get_args():
	descr = "Plot aggregated time data from the hyperparameter optimization"
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('hpo_dir', default='hyperopt')
	parser.add_argument('--cache-logs', nargs=2)
	parser.add_argument('--raw_times', action='store_true')
	return parser.parse_args()

def collect_time_data(args):

	data = dict(
		time     = defaultdict(lambda: 0.0),
		raw_time = defaultdict(lambda: 0.0)
	)

	NAMES = ['nmn', 'encoder', 'find', 'measure', 'describe']
	for name in NAMES:
		fn_pattern = os.path.join(args.hpo_dir, name, name+'-*_log.json')
		logfiles   = glob(fn_pattern)
		for fn in logfiles:
			with open(fn) as fd:
				log = json.load(fd)
			for k in ['time', 'raw_time']:
				data[k][name] += log[k][-1]

	for fn in args.cache_logs:
		with open(fn) as fd:
			log = json.load(fd)
		for k in ['time', 'raw_time']:
			data[k]['cache'] += log[k]

	return data

def plot_data(args, data):

	suf = '(raw)' if args.raw_times else '(forward-backward)'
	plt.figure()
	sns.set_palette(sns.color_palette("Dark2"))
	plt.title('Training times '+suf)
	plt.ylabel('Hours')

	bottom = 0
	width = 0.5
	plots = list()

	time_key = 'raw_time' if args.raw_times else 'time'

	NAMES = ['encoder', 'find', 'cache', 'measure', 'describe']
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
	args = get_args()
	data = collect_time_data(args)
	plot_data(args, data)