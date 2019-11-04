import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

VALUE_KEY  = 'top_1'
MIN_VALUE  = 0.01
N_VALUES   = 30
N_SLOTS    = 10

if __name__ == '__main__':

	glob_list = glob('find-rnd/find-rnd*.json')
	value_list = list()
	fn_list = list()
	for fn in glob_list:
		with open(fn) as fd:
			v = json.load(fd)[VALUE_KEY]
		if v > MIN_VALUE:
			value_list.append(v)
			fn_list.append(fn)

	N_PER_SLOT = int(N_VALUES/N_SLOTS)
	min_v = min(value_list)
	max_v = max(value_list)
	displ = -min_v
	scale = 1./(max_v-min_v+1e-10)

	n_list = [0]*N_SLOTS
	sel_values = list()
	sel_filenames = list()
	sel_indices   = set()
	while len(sel_values) < N_VALUES and N_PER_SLOT < N_VALUES/2:
		for idx, (v, fn) in enumerate(zip(value_list, fn_list)):
			i = int(len(n_list)*(v+displ)*scale)
			if n_list[i] < N_PER_SLOT and idx not in sel_indices:
				n_list[i] += 1
				sel_indices.add(idx)
				sel_values.append(v)
				sel_filenames.append(fn)
		N_PER_SLOT += 1

	print('{} points to {}'.format(len(value_list), len(sel_values)))

	plt.figure()
	plt.hist(sel_values)
	plt.show()

	plt.figure()
	plt.scatter(value_list, [0.5]*len(value_list), c='grey', alpha=0.5)
	plt.scatter(sel_values, [1]*len(sel_values), alpha=0.5)
	plt.show()

	if input('Write to JSON file? ').lower() in {'y','yes'}:
		l = len('json')
		pt_filenames = [ os.path.basename(fn[:-l]+'pt') for fn in sel_filenames ]
		data = dict(
			pt_files   = pt_filenames,
			gauge_accs = sel_values,
			nmn_accs   = [-1]*len(sel_values)
		)
		with open('gauge_corr_data.json', 'w') as fd:
			json.dump(data, fd)

