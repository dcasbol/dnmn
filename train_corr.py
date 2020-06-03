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

	if os.path.exists(args.hpo_res):
		with open(args.hpo_res, 'rb') as fd:
			res = pickle.load(fd)
		conf = res.x_iters[res.best_eval]
	else:
		print('No hpo-res file. Using default configuration.')
		conf = dict(
			batch_size    = 356,
			learning_rate = 0.0007700193882725058,
			dropout       = 0.15733753600629702,
			weight_decay  = 1.2561860220574558e-06
		)

	runner = NMNRunner(
		max_epochs    = 50,
		validate      = True,
		batch_size    = int(conf['batch_size']),
		learning_rate = conf['learning_rate'],
		dropout       = conf['dropout'],
		weight_decay  = conf['weight_decay'],
		find_pt       = pt_file
	)
	runner.run()

	d = read()
	d['nmn_accs'][idx]     = runner.best_acc
	d['nmn_rel_accs'][idx] = max(runner._logger._log['rel_acc'])
	d['nmn_loss'][idx]     = min(runner._logger._log['val_loss'])
	write(d)

	print('Best validation accuracy:', runner.best_acc)
	print('Achieved at epoch', runner.best_epoch)

if __name__ == '__main__':
	main(get_args())