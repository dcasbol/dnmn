import os
import argparse
import json
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt

def get_args():
	descr = "Plot aggregated time data from the hyperparameter optimization"
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('hpo_dir', default='hyperopt')
	parser.add_argument('--raw-times', action='store_true')
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

	return data

def plot_data(args, data):

	plt.figure()
	plt.ylabel('Hours')

	palette = {
		'find'       : 'blue',
		'measure'    : 'orange',
		'describe'   : 'green',
		'encoder'    : 'red',
		'end-to-end' : 'purple'
	}

	bottom = 0
	width  = 0.5
	plots  = list()

	time_key = 'raw_time' if args.raw_times else 'time'

	NAMES = ['encoder', 'find', 'measure', 'describe']
	for name in NAMES:
		t = data[time_key][name]/3600
		p = plt.bar(0, t, width, bottom=bottom, color=palette[name])
		plots.append(p)
		bottom += t

	t = data[time_key]['nmn']/3600
	plt.bar(1, t, width, color=palette['end-to-end'])

	plt.xticks((0, 1), ('modular\n(aggregated cost)', 'end-to-end'))
	plots.reverse()
	plt.legend(plots, [ name.capitalize() for name in reversed(NAMES) ])

	plt.show()

if __name__ == '__main__':
	args = get_args()
	data = collect_time_data(args)
	plot_data(args, data)