import os
import argparse
import skopt
from runners.runners import EncoderRunner, FindRunner, MeasureRunner, DescribeRunner
from runners.runners import NMNRunner

def get_args():
	parser = argparse.ArgumentParser(description='Hyperparameter optimization')
	parser.add_argument('selection', choices=['find', 'describe', 'measure', 'encoder', 'nmn'])
	return parser.parse_args()

SPACE = [
	skopt.space.Integer(16, 512, name='batch_size'),
	skopt.space.Real(1e-5, 0.1, name='learning_rate', prior='log-uniform'),
	skopt.space.Real(0, 0.9, name='dropout'),
	skopt.space.Real(1e-5, 1., name='weight_decay', prior='log-uniform')
]

class HyperOptimizer(object):

	def __init__(self, selection, best_pt):
		self._sel = selection
		self._path_dir = 'hyperopt/{}'.format(selection)
		self._path_res = '{}/{}-res.gz'.format(self._path_dir, selection)
		self._num_evals = 0
		self._best_acc = 0.
		self._best_pt = best_pt

		if not os.path.exists(self._path_dir):
			os.makedirs(self._path_dir)

		self._runner_cl = dict(
			encoder  = EncoderRunner,
			find     = FindRunner,
			measure  = MeasureRunner,
			describe = DescribeRunner,
			nmn      = NMNRunner
		)[self._sel]

		self._res = None
		self._x0 = self._y0 = None
		if os.path.exists(self._path_res):
			print('Found previous skopt result. Resuming.')
			self._res = skopt.load(self._path_res)
			self._x0 = self._res.x_iters
			self._y0 = self._res.func_vals
			self._best_acc = -self._res.fun
			self._num_evals = len(self._y0)

	def _eval(self, batch_size, learning_rate, dropout, weight_decay):

		self._num_evals += 1

		batch_size = int(batch_size) # Numpy int is problematic

		suffix = 'hpo({n:02d})-bs{bs}-lr{lr:.2g}-{do:.1f}do-wd{wd:.2g}'.format(
			n   = self._num_evals-1,
			bs  = batch_size,
			lr  = learning_rate,
			do  = dropout,
			wd  = weight_decay
		)

		test = self._runner_cl(
			max_epochs    = 30,
			batch_size    = batch_size,
			learning_rate = learning_rate,
			dropout       = dropout,
			weight_decay  = weight_decay,
			suffix        = suffix,
			validate      = True
		)
		test.run()

		res_suffix = '{}-bep{}'.format(suffix, test.best_epoch)

		if test.bet_acc > self._best_acc:
			test.save_model(self._best_pt)

		self._best_acc = max(self._best_acc, test.best_acc)
		print('Eval({}): {:.1f}-{}'.format(self._num_evals, test.best_acc, res_suffix))
		print('Best acc is', self._best_acc)

		json_fn = '{}-{}_log.json'.format(self._sel, suffix)
		new_fn  = '{}-{:05.1f}-{}_log.json'.format(self._sel, test.best_acc, suffix)
		new_fn  = os.path.join(self._path_dir, new_fn)
		os.rename(json_fn, new_fn)

		return -test.best_acc

	def _save(self, res):
		args = res.specs['args']
		if 'func' in args:
			del args['func']
		if 'callback' in args:
			del args['callback']
		skopt.dump(res, self._path_res)

	def run(self):

		@skopt.utils.use_named_args(SPACE)
		def obj_func(**kwargs):
			return self._eval(**kwargs)

		def callback(res):
			return self._save(res)

		skopt.gp_minimize(obj_func, SPACE,
			verbose = True,
			x0 = self._x0, y0 = self._y0,
			callback = callback,
			n_random_starts = 5,
			n_calls = 20
		)

		print('Hyperparameter Optimization ended.')
		print('Best result:', self._best_acc)

if __name__ == '__main__':

	args = get_args()
	opt = HyperOptimizer(args.selection)
	opt.run()
	
