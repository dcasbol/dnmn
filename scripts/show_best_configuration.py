import skopt
import argparse

def get_args():
	descr = """Read a skopt result file and print the best configuration"""
	parser = argparse.ArgumentParser(description=descr)
	parser.add_argument('result_file')
	return parser.parse_args()

PARAM_NAMES = ['batch_size', 'learning_rate', 'dropout', 'weight_decay']

def main(args):

	print('Showing results in {!r}...', args.result_file)

	res = skopt.load(args.result_file)
	print('Final accuracy: {}%'.format(-100*res.fun))
	print('Hyperparameters:')
	for name, value in zip(PARAM_NAMES, res.x):
		print(name, ' --> ', value)

if __name__ == '__main__':
	main(get_args())
	