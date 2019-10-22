
import argparse
import json

NAMES = ['nmn', 'encoder', 'find', 'measure', 'describe']

def get_args():
	descr = """Collect all training times in a single JSON file."""
	parser = argparse.ArgumentParser(description=descr)
	for name in NAMES:
		parser.add_argument('--'+name)
	parser.add_argument('--cache')
	parser.add_argument('--output-log', default='training-times.json')
	return parser.parse_args()

def main(args):

	data = dict(
		time     = dict(),
		raw_time = dict()
	)

	for name in NAMES:
		fn = getattr(args, name)
		assert fn is not None, 'Logfile for {} was not specified'.format(name)
		with open(fn) as fd:
			d = json.load(fd)
		data['time'][name]     = d['time'][-1]
		data['raw_time'][name] = d['raw_time'][-1]

	assert args.cache is not None, 'Caching logfile was not specified'
	with open(args.cache) as fd:
		d = json.load(fd)
	data['time']['cache']     = d['time']
	data['raw_time']['cache'] = d['raw_time']

	with open(args.output_log, 'w') as fd:
		json.dump(data, fd)

if __name__ == '__main__':
	main(get_args())
