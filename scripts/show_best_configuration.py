import json
import pickle
import argparse

class ResultObject:
	pass

def get_args():
	descr = """Read a skopt result file and print the best configuration"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('result_file')
	return parser.parse_args()

PARAM_NAMES = ['batch_size', 'learning_rate', 'dropout', 'weight_decay']

def main(args):

	print('Showing results in {!r}...'.format(args.result_file))

	with open(args.result_file, 'rb') as fd:
		res = pickle.load(fd)

	print('Final accuracy: {}%'.format(res.best_acc))
	print('Hyperparameters:')
	for name, value in res.x_iters[res.best_eval].items():
		print(name, ' --> ', value)

if __name__ == '__main__':
	main(get_args())
