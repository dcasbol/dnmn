import os
import json
import argparse
import matplotlib.pyplot as plt
from glob import glob

MIN_VALUE  = 0.01
N_SLOTS    = 10

def get_args():
	descr = """Create JSON with selected Find modules for correlation plot"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('--input-pattern', default='find-rnd/find-rnd*.json')
	# 2*std < 0.1 --> std < 0.05 --> var < 2.5e-3
	parser.add_argument('--max-variance', type=float, default=2.5e-3)
	parser.add_argument('--n-values', type=int, default=30)
	return parser.parse_args()

def main():

	args = get_args()

	glob_list  = glob(args.input_pattern)
	var_list   = list()
	value_list = list()
	fn_list    = list()
	skipped_acc = skipped_var = 0
	for fn in glob_list:
		with open(fn) as fd:
			d = json.load(fd)
		if d['top_1'] < MIN_VALUE:
			skipped_acc += 1
			continue
		if d.get('var', 0) > args.max_variance:
			skipped_var += 1
			continue

		value_list.append(d['top_1'])
		fn_list.append(fn)
		if 'var' in d:
			var_list.append(d['var'])

	for n, reason in [(skipped_acc, 'accuracy'), (skipped_var, 'variance')]:
		if n == 0: continue
		print('{} values skipped due to {} constraints'.format(n, reason))

	# Sort by variance
	if len(var_list) > 0:
		indices = sorted(list(range(len(value_list))), key=lambda i: var_list[i])
		value_list = [ value_list[i] for i in indices ]
		fn_list = [ fn_list[i] for i in indices ]
		var_list = [ var_list[i] for i in indices ]


	N_VALUES = args.n_values
	N_PER_SLOT = int(N_VALUES/N_SLOTS)
	min_v = min(value_list)
	max_v = max(value_list)
	displ = -min_v
	scale = 1./(max_v-min_v+1e-10)

	n_list = [0]*N_SLOTS
	sel_values = list()
	sel_filenames = list()
	sel_indices   = set()
	while len(sel_values) < N_VALUES and N_PER_SLOT < N_VALUES:
		for idx, (v, fn) in enumerate(zip(value_list, fn_list)):
			i = int(len(n_list)*(v+displ)*scale)
			if n_list[i] < N_PER_SLOT and idx not in sel_indices:
				n_list[i] += 1
				sel_indices.add(idx)
				sel_values.append(v)
				sel_filenames.append(fn)
		N_PER_SLOT += 1

	print('{} selected from {}'.format(len(sel_values), len(value_list)))

	plt.figure()
	plt.hist(sel_values)
	plt.show()

	plt.figure()
	if len(var_list) == 0:
		plt.scatter(value_list, [0.5]*len(value_list), c='grey', alpha=0.5)
		plt.scatter(sel_values, [1]*len(sel_values), alpha=0.5)
	else:
		excl_indices = [ i for i in range(len(value_list)) if i not in sel_indices ]
		excl_values = [ value_list[i] for i in excl_indices ]
		excl_vars   = [ var_list[i] for i in excl_indices ]
		plt.scatter(excl_values, excl_vars, c='grey', alpha=0.5)

		sel_vars = [ var_list[i] for i in sel_indices ]
		plt.scatter(sel_values, sel_vars, alpha=0.5)
	plt.show()

	if input('Write to JSON file? ').lower() in {'y','yes'}:
		l = len('json')
		pt_filenames = [ os.path.basename(fn[:-l]+'pt') for fn in sel_filenames ]
		data = dict(
			pt_files   = pt_filenames,
			gauge_accs = sel_values,
			nmn_accs   = [-1]*len(sel_values)
		)
		if len(var_list) > 0:
			data['gauge_vars'] = sel_vars
		with open('gauge_corr_data.json', 'w') as fd:
			json.dump(data, fd)

if __name__ == '__main__':
	main()
