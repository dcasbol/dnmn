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
	parser.add_argument('--max-std', type=float, default=0.005)
	parser.add_argument('--n-values', type=int, default=30)
	return parser.parse_args()

def main(args):

	glob_list  = glob(args.input_pattern)

	max_variance = args.max_std**2

	var_list   = list()
	value_list = list()
	fn_list    = list()
	agr_list   = list()
	skipped_acc = skipped_var = 0
	for fn in glob_list:
		with open(fn) as fd:
			d = json.load(fd)

		skip_acc = d['top_1'] < MIN_VALUE
		skip_var = d['var'] > max_variance
		if skip_acc or skip_var:
			skipped_acc += int(skip_acc)
			skipped_var += int(skip_var)
			continue

		value_list.append(d['top_1'])
		var_list.append(d['var'])
		fn_list.append(fn)
		agr_list.append(d.get('agreement',0))

	for n, reason in [(skipped_acc, 'accuracy'), (skipped_var, 'variance')]:
		if n == 0: continue
		print('{} values violated {} constraints'.format(n, reason))
	if skipped_acc + skipped_var > 0:
		print('{} skipped values in total'.format(len(glob_list)-len(value_list)))

	# Sort by variance
	select = lambda l, idcs: [ l[i] for i in idcs ]
	indices = sorted(range(len(value_list)), key=lambda i: var_list[i])
	value_list = select(value_list, indices)
	fn_list    = select(fn_list, indices)
	var_list   = select(var_list, indices)
	agr_list   = select(agr_list, indices)

	N_VALUES = args.n_values
	min_v = min(value_list)
	max_v = max(value_list)
	displ = -min_v
	scale = 1./(max_v-min_v+1e-10)
	get_slot = lambda v: int(N_SLOTS*(v+displ)*scale)

	slots = [ list() for i in range(N_SLOTS) ]
	for idx, v in enumerate(value_list):
		slots[get_slot(v)].append(idx)

	i = 0
	sel_indices = list()
	while len(sel_indices) < N_VALUES:
		for idcs in slots:
			if i < len(idcs):
				sel_indices.append(idcs[i])
		i += 1
	sel_values  = [ value_list[i] for i in sel_indices ]
	sel_filenames = [ fn_list[i] for i in sel_indices ]

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

		plt.figure()
		sel_agrs = select(agr_list, indices)
		plt.scatter(sel_values, sel_agrs, alpha=0.5)
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
	main(get_args())
