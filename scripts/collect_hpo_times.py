import os
import argparse
import json
from glob import glob
from collections import defaultdict

NAMES = ['nmn', 'encoder', 'find', 'measure', 'describe']

def get_args():
	descr = """Collect all time data from the hyperparameter optimization
	in a single JSON file."""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('hpo_dir', default='hyperopt')
	parser.add_argument('--cache-logs', nargs=2)
	parser.add_argument('--output-log', default='hpo-times.json')
	return parser.parse_args()

def main(args):

	data = dict(
		time     = defaultdict(lambda: 0.0),
		raw_time = defaultdict(lambda: 0.0)
	)

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

	with open(args.output_log, 'w') as fd:
		json.dump(data, fd)

if __name__ == '__main__':
	main(get_args())
