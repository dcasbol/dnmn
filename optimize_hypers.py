import os
import argparse
import skopt
from hypers.runners import EncoderRunner, FindRunner, MeasureRunner, DescribeRunner #NMNRunner

def get_args():
	parser = argparse.ArgumentParser(description='Hyperparameter optimization')
	parser.add_argument('selection', choices=['find', 'describe', 'measure', 'encoder', 'nmn'])
	return parser.parse_args()

SPACE = [
	skopt.space.Integer(4, 10, name='batch_size_exp'),
	skopt.space.Real(1e-5, 0.1, name='learning_rate', prior='log-uniform'),
	skopt.space.Integer(0, 1, name='dropout_int'),
	skopt.space.Integer(0, 4, name='embed_size_idx'),
	skopt.space.Real(1e-5, 1., name='weight_decay', prior='log-uniform')
]

class HyperOptimizer(object):

	def __init__(self, selection):
		self._sel = selection
		self._path_dir = 'hyperopt/{}'.format(selection)
		self._path_res = '{}/{}-res.gz'.format(self._path_dir, selection)
		self._num_evals = 0
		self._best_acc = 0.

		if not os.path.exists(self._path_dir):
			os.makedirs(self._path_dir)

		self._runner_cl = dict(
			encoder  = EncoderRunner,
			find     = FindRunner,
			measure  = MeasureRunner,
			describe = DescribeRunner,
			#nmn      = NMNRunner
		)[self._sel]

		self._res = None
		self._x0 = self._y0 = None
		if os.path.exists(self._path_res):
			print('Found previous skopt result. Resuming.')
			self._res = skopt.load(self._path_res)
			self._x0 = self._res.x_iters
			self._y0 = self._res.func_vals

	def _eval(self, batch_size_exp, learning_rate, dropout_int, embed_size_idx, weight_decay):

		self._num_evals += 1

		batch_size = int(2**batch_size_exp)
		dropout = bool(dropout_int)
		embed_size = [60, 125, 250, 500, 1000][embed_size_idx]

		suffix = 'hpo-bs{bs}-lr{lr:.2g}-{do}do-emb{emb}-wd{wd:.2g}'.format(
			bs  = batch_size,
			lr  = learning_rate,
			do  = dropout_int*5,
			emb = embed_size,
			wd  = weight_decay
		)

		test = self._runner_cl(
			batch_size    = batch_size,
			learning_rate = learning_rate,
			dropout       = dropout,
			embed_size    = embed_size,
			weight_decay  = weight_decay,
			suffix        = suffix,
			validate      = True
		)
		test.run()

		res_suffix = '{}-ep{}'.format(suffix, test.best_epoch)

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
		if self._num_evals == 1:
			skopt.load(self._path_res)
			print('Saved OK')

	def run(self):

		@skopt.utils.use_named_args(SPACE)
		def obj_func(**kwargs):
			return self._eval(**kwargs)

		def callback(res):
			return self._save(res)

		skopt.gp_minimize(obj_func, SPACE,
			verbose = True,
			x0 = self._x0, y0 = self._y0,
			callback = callback
		)

		print('Hyperparameter Optimization ended.')

if __name__ == '__main__':

	args = get_args()
	opt = HyperOptimizer(args.selection)
	opt.run()
	
