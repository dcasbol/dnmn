import os
import re
import json
import argparse
import numpy as np
from glob import glob

def get_args():
	parser = argparse.ArgumentParser(description='Generate plot info for assessing Find utility.')
	parser.add_argument('--describe-logs-dir', default='./')
	parser.add_argument('--output-log', default='plotqual.json')
	parser.add_argument('--whiten', action='store_true')
	parser.add_argument('--prefix', default='describe-qual-ep-')
	return parser.parse_args()

def main(args):

	i0 = len(args.prefix)

	pattern = os.path.join(args.describe_logs_dir, '%s*_log.json' % args.prefix)
	fn_list = glob(pattern)
	fn_list.sort()

	acc_list = list()
	epoch_list = list()
	max_epoch_list = list()

	for fn in fn_list:
		with open(fn) as fd:
			data = json.load(fd)
		if len(data['epoch']) < 4: continue
		max_acc = max(data['top_1'])
		if data['top_1'][-1] < max_acc:
			acc_list.append(max_acc)
			max_epoch_list.append(data['epoch'][-4])

			basename = os.path.basename(fn)[i0:]
			m = re.search('^(\d+(\.\d+)?)', basename)
			epoch = float(m.group(1))
			epoch_list.append(epoch)

	if len(epoch_list) == 0: return

	result = dict(
		epoch = epoch_list,
		top_1 = acc_list,
		max_epoch = max_epoch_list
	)

	if args.whiten:
		for k in [ k for k in result.keys() if k != 'epoch' ]:
			values = result[k]
			new_values = (np.array(values) - np.mean(values)) / np.std(values)
			result[k] = new_values.tolist()

	with open(args.output_log, 'w') as fd:
		json.dump(result, fd)


if __name__ == '__main__':
	main(get_args())
