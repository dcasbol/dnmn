import os
import argparse
import pickle
import json
from runners import EncoderRunner, FindRunner, MeasureRunner, DescribeRunner
from runners import NMNRunner

def get_args():
	parser = argparse.ArgumentParser(description='Hyperparameter optimization')
	parser.add_argument('selection', choices=['find', 'describe', 'measure', 'encoder', 'nmn'])
	parser.add_argument('--target-dir', default=None)
	parser.add_argument('--candidates', default='hyperopt/hpo_candidates.json')
	parser.add_argument('--modular', action='store_true')
	return parser.parse_args()

def read_pickle(file_name):
	with open(file_name, 'rb') as fd:
		data = pickle.load(fd)
	return data

def write_pickle(file_name, data):
	with open(file_name, 'wb') as fd:
		pickle.dump(data, fd)

class ResultObject:
	pass

class HyperOptimizer(object):

	def __init__(self, selection, modular, path_candidates, target_dir=None):

		if modular and selection == 'encoder':
			print("Modular flag doesn't affect training of Question Encoder.")
			print("Flag ignored.")

		self._sel = selection
		self._modular = modular
		self._path_candidates = path_candidates
		self._base_dir = target_dir
		if target_dir is None:
			self._base_dir = 'hyperopt/' + 'modular/'*int(modular)
		self._path_dir = os.path.join(self._base_dir, selection)
		self._path_res = '{}/{}-res.dat'.format(self._path_dir, selection)
		self._eval_idx = 0
		self._best_pt  = '{}/{}-hpo-best.pt'.format(self._path_dir, selection)
		self._best_acc = None
		self._test_obj = None

		if not os.path.exists(self._path_dir):
			os.makedirs(self._path_dir)

		with open(self._path_candidates) as fd:
			self._candidates = json.load(fd)

		self._runner_cl = dict(
			encoder  = EncoderRunner,
			find     = FindRunner,
			measure  = MeasureRunner,
			describe = DescribeRunner,
			nmn      = NMNRunner
		)[self._sel]

		self._res = ResultObject()
		self._res.x_iters   = list()
		self._res.func_vals = list()
		self._res.best_acc  = 0
		self._res.best_eval = -1

		if os.path.exists(self._path_res):
			print('Found previous HPO file. Resuming optimization.')
			self._res = read_pickle(self._path_res)
			self._best_acc = self._res.best_acc
			self._eval_idx = len(self._res.func_vals)
			assert self._eval_idx < len(self._candidates),\
				"Can't resume. Max. calls already reached."

	def _eval(self, batch_size, learning_rate, dropout, weight_decay):

		suffix = 'hpo({n:02d})-bs{bs}-lr{lr:.2g}-{do:.1f}do-wd{wd:.2g}'.format(
			n   = self._eval_idx,
			bs  = batch_size,
			lr  = learning_rate,
			do  = dropout,
			wd  = weight_decay
		)

		kwargs = {'modular':self._modular} if self._sel != 'encoder' else {}
		test = self._runner_cl(
			max_epochs    = 50,
			batch_size    = batch_size,
			learning_rate = learning_rate,
			dropout       = dropout,
			weight_decay  = weight_decay,
			suffix        = suffix,
			validate      = True,
			**kwargs
		)
		test.run()

		if self._best_acc is None or test.best_acc > self._best_acc:
			self._test_obj = test
			self._best_acc = test.best_acc
		else:
			self._test_obj = None

		res_suffix = '{}-bep{}'.format(suffix, test.best_epoch)

		print('Eval({}): {:.1f}-{}'.format(self._eval_idx, test.best_acc, res_suffix))
		print('Best HPO acc is', self._best_acc)

		modname = self._sel if self._sel != 'find' else 'gauge-find'
		json_fn = '{}-{}_log.json'.format(modname, suffix)
		new_fn  = '{}-{:05.1f}-{}_log.json'.format(self._sel, test.best_acc, suffix)
		new_fn  = os.path.join(self._path_dir, new_fn)
		os.rename(json_fn, new_fn)

		return test.best_acc

	def run(self):

		for i in range(self._eval_idx, len(self._candidates)):
			print('Evaluation', i)
			self._eval_idx = i

			acc = self._eval(**self._candidates[i])

			self._res.x_iters.append(self._candidates[i])
			self._res.func_vals.append(acc)
			if acc > self._res.best_acc:
				self._res.best_acc  = acc
				self._res.best_eval = i

			write_pickle(self._path_res, self._res)
			if self._test_obj is not None:
				self._test_obj.save_model(self._best_pt)
				self._test_obj = None

		print('Hyperparameter Optimization ended.')
		print('Best result:', self._res.best_acc)
		print('Found at eval. index:', self._res.best_eval)
		print('Best hyperparameters:')
		print(self._candidates[self._res.best_eval])

if __name__ == '__main__':

	args = get_args()
	opt = HyperOptimizer(args.selection, args.modular, args.candidates,
		target_dir=args.target_dir)
	opt.run()
	
