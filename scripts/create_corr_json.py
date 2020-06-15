import os
import json
import argparse
import matplotlib.pyplot as plt
from glob import glob

N_SLOTS    = 20

def get_args():
	descr = """Create JSON with selected Find modules for correlation plot"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('--input-pattern', default='find-rnd/find-rnd-*.json')
	parser.add_argument('--max-std', type=float, default=0.05)
	parser.add_argument('--n-values', type=int, default=30)
	parser.add_argument('--select-by', choices=['top_1','rel_acc','val_loss'], default='val_loss')
	parser.add_argument('--silent', action='store_true')
	parser.add_argument('--force-save', action='store_true')
	return parser.parse_args()

def extract(dict_list, key):
	return [ d[key] for d in dict_list ]

def main(args):

	glob_list  = glob(args.input_pattern)

	max_variance = args.max_std**2

	selected = list()
	discarded = list()
	for fn in glob_list:
		with open(fn) as fd:
			d = json.load(fd)
		d['fn'] = fn
		if d['var'] > max_variance:
			discarded.append(d)
		else:
			selected.append(d)

	if len(discarded) > 0:
		print('{} values violated variance constraints'.format(len(discarded)))

	# Sort by variance
	selected = sorted(selected, key=lambda s: s['var'])

	N_VALUES = args.n_values
	value_list = extract(selected, args.select_by)
	min_v = min(value_list)
	max_v = max(value_list)
	displ = -min_v
	scale = 1./(max_v-min_v+1e-10)
	get_slot = lambda v: int(N_SLOTS*(v+displ)*scale)

	slots = [ list() for i in range(N_SLOTS) ]
	for s in selected:
		slots[get_slot(s[args.select_by])].append(s)

	i = 0
	final = list()
	while len(final) < N_VALUES:
		for sl in slots:
			if i < len(sl):
				final.append(sl[i])
		i += 1
		assert i < 10000, 'Ran out of iterations, maybe not enough points to work with.'

	print('{} selected from {}'.format(len(final), len(selected)))

	# Selection distribution
	if not args.silent:
		plt.figure()
		plt.hist([ s[args.select_by] for s in final ])
		plt.show()

	# Show all discarded and selected
	if not args.silent:
		xlabel = dict(
			top_1    = 'Accuracy',
			rel_acc  = 'Relative Accuracy',
			val_loss = 'Validation Loss'
		)[args.select_by]
		selected = [ s for s in selected if s not in final ]
		plt.figure()
		plt.scatter(extract(discarded, args.select_by), extract(discarded, 'var'),
			c='red', alpha=0.25)
		plt.scatter(extract(selected, args.select_by), extract(selected, 'var'),
			c='grey', alpha=0.5)
		plt.scatter(extract(final, args.select_by), extract(final, 'var'),
			c='blue', alpha=0.5)
		plt.ylabel('Predictive Variance')
		plt.xlabel(xlabel)
		plt.show()

	if args.force_save or input('Write to JSON file? ').lower() in {'y','yes'}:
		l = len('json')
		pt_filenames = [ os.path.basename(fn[:-l]+'pt') for fn in extract(final, 'fn') ]
		data = dict(
			pt_files       = pt_filenames,
			gauge_accs     = extract(final, 'top_1'),
			gauge_rel_accs = extract(final, 'rel_acc'),
			gauge_loss     = extract(final, 'val_loss'),
			gauge_vars     = extract(final, 'var')
		)
		for k in ['nmn_accs','nmn_rel_accs','nmn_loss']:
			data[k] = [-1]*len(final)
		with open('gauge_corr_data.json', 'w') as fd:
			json.dump(data, fd)
	else:
		print('JSON discarded')

if __name__ == '__main__':
	main(get_args())
