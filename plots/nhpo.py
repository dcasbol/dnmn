import argparse
import json
import glob
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
	parser = argparse.ArgumentParser(
		description='Robustness modular vs. end-to-end training')
	parser.add_argument('base_dir')
	parser.add_argument('--accuracies', action='store_true')
	parser.add_argument('--times', action='store_true')
	return parser.parse_args()

def collate_accuracies(dir_list, version):
	basename = '{}-results/OpenEnded_mscoco_val2014_accuracy.json'.format(version)
	accs = list()
	for d in dir_list:
		fn = os.path.join(d, basename)
		try:
			with open(fn) as fd:
				a = json.load(fd)['overall']
				accs.append(a)
		except FileNotFoundError:
			continue
	return accs

def collate_times(dir_list, name):
	basepat = '{0}/{0}-*-hpo*_log.json'.format(name)
	times = list()
	for d in dir_list:
		t = 0
		pat = os.path.join(d, basepat)
		for fn in glob.glob(pat):
			with open(fn) as fd:
				t += json.load(fd)['raw_time'][-1]
		times.append(t/3600)
	return times

def main(args):

	assert args.accuracies or args.times, 'Select plot options'

	pat1 = os.path.join(args.base_dir, 'run-[123456789]')
	pat2 = os.path.join(args.base_dir, 'run-[123456789][0123456789]')
	dir_list = sorted(glob.glob(pat1)) + sorted(glob.glob(pat2))

	if args.accuracies:
		modular_accs = collate_accuracies(dir_list, 'modular')
		end2end_accs = collate_accuracies(dir_list, 'nmn')
		data  = [ {'modality':'modular',    'accuracy':a} for a in modular_accs ]
		data += [ {'modality':'end-to-end', 'accuracy':a} for a in end2end_accs ]
		data  = pd.DataFrame(data)

		sns.violinplot(x='accuracy', y='modality', data=data, inner=None)
		sns.swarmplot(x='accuracy', y='modality', data=data, color='black')
		plt.show()

	if args.times:
		MODULE_NAMES  = ['encoder', 'find', 'describe', 'measure']
		modular_times = [ collate_times(dir_list, name) for name in MODULE_NAMES ]
		modular_times = [ sum(times) for times in zip(*modular_times) ]
		end2end_times = collate_times(dir_list, 'nmn')
		data  = [ {'modality':'modular',    'hours':a} for a in modular_times ]
		data += [ {'modality':'end-to-end', 'hours':a} for a in end2end_times ]
		data  = pd.DataFrame(data)

		sns.violinplot(x='hours', y='modality', data=data, inner=None)
		sns.swarmplot(x='hours', y='modality', data=data, color='black')
		plt.show()

if __name__ == '__main__':
	main(get_args())