import argparse
import skopt
import pdb
import pickle
from hypers.runners import EncoderRunner

def get_args():
	parser = argparse.ArgumentParser(description='Hyperparameter optimization')
	parser.add_argument('module', choices=['find', 'describe', 'measure', 'encoder'])
	return parser.parse_args()

SPACE = [
	skopt.space.Real(16, 1024, name='batch_size_float', prior='log-uniform'),
	skopt.space.Real(1e-5, 0.1, name='learning_rate', prior='log-uniform'),
	skopt.space.Integer(0, 1, name='dropout_int'),
	skopt.space.Integer(0, 4, name='embed_size_idx')
]

@skopt.utils.use_named_args(SPACE)
def test_encoder(batch_size_float, learning_rate, dropout_int, embed_size_idx):

	batch_size = int(batch_size_float + 0.5)
	dropout = bool(dropout_int)
	embed_size = [60, 125, 250, 500, 1000][embed_size_idx]

	test = EncoderRunner(
		batch_size    = batch_size,
		learning_rate = learning_rate,
		dropout       = dropout,
		embed_size    = embed_size,
		validate      = True
	)
	test.run()
	return -test.best_acc


if __name__ == '__main__':

	result = skopt.forest_minimize(test_encoder, SPACE, acq_func='EI')
	skopt.dump(result, 'result_hypers.gz')