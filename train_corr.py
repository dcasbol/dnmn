import os
import argparse
import json
import pickle
from runners import NMNRunner


class ResultObject:
	pass

def get_args(args=None):
	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('--corr-data', default='gauge_corr_data.json')
	parser.add_argument('--find-pt-dir', default='find-rnd')
	parser.add_argument('--hpo-res', default='hyperopt/nmn/nmn-res.dat')
	return parser.parse_args(args)

def main(args):

	def read():
		with open(args.corr_data) as fd:
			return json.load(fd)

	def write(d):
		with open(args.corr_data, 'w') as fd:
			json.dump(d, fd)

	d = read()

	for i in range(len(d['pt_files'])):
		if d['nmn_accs'][i] == -1:
			d['nmn_accs'][i] = -2
			write(d)
			idx = i
			pt_file = os.path.join(args.find_pt_dir, d['pt_files'][i])
			break
	else:
		print('No more jobs left')
		quit()

	with open(args.hpo_res, 'rb') as fd:
		res = pickle.load(fd)
	conf = res.x_iters[res.best_eval]

	runner = NMNRunner(
		max_epochs    = 100,
		validate      = True,
		batch_size    = int(conf['batch_size']),
		learning_rate = conf['learning_rate'],
		dropout       = conf['dropout'],
		weight_decay  = conf['weight_decay'],
		find_pt       = pt_file
	)
	runner.run()

	d = read()
	d['nmn_accs'][idx] = runner.best_acc
	write(d)

	print('Best validation accuracy:', runner.best_acc)
	print('Achieved at epoch', runner.best_epoch)

if __name__ == '__main__':
	main(get_args())