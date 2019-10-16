import os
import json
import argparse
import numpy as np
from glob import glob

def get_args():
	descr = """Generate json for checking correlation between gauge module's accuracy
	and the final NMN accuracy, when using that module"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('gauge-find-log')
	parser.add_argument('--logs-dir', default='./')
	parser.add_argument('--output-log', default='merged_max_acc.json')
	parser.add_argument('--whiten', action='store_true')
	parser.add_argument('--prefix', default='nmn-find_corr')
	return parser.parse_args()

def main(args):

	i0 = len(args.prefix)

	pattern_base = os.path.join(args.logs_dir, args.prefix)

	filename_list = glob(pattern_base+'-ep*-run0_log.json')
	n = len('00-run0_log.json')
	epoch_list = [ int(fn[-n:-n+2]) for fn in filename_list ]
	epoch_list.sort()
	n_epochs = len(epoch_list)
	n_runs   = len(glob(pattern_base+'-ep00-run*_log.json'))

	top_1_means = list()
	top_1_stds  = list()
	for i, epoch in enumerate(range(epoch_list)):
		top_1_list = list()
		for run in range(n_runs):
			fn = pattern_base+'ep{:02d}-run{}_log.json'.format(epoch, run)
			with open(fn) as fd:
				data = json.load(fd)
			top_1_list.append(max(data['top_1']))
		top_1_means.append(np.mean(top_1_list))
		top_1_stds.append(np.std(top_1_list))

	with open(args.gauge_find_log) as fd:
		data = json.load(fd)
	find_top_1_list = [ data['top_1'][epoch] for epoch in epoch_list ]
	find_top_1_train_list = [ data['top_1_train'][epoch] for epoch in epoch_list ]

	result = dict(
		epoch            = epoch_list,
		nmn_top_1_mean   = top_1_means,
		nmn_top_1_std    = top_1_stds,
		find_top_1       = find_top_1_list,
		find_top_1_train = find_top_1_train_list
	)

	with open(args.output_log, 'w') as fd:
		json.dump(result, fd)


if __name__ == '__main__':
	main(get_args())
