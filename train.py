import argparse
from runners.runners import EncoderRunner, FindRunner, MeasureRunner, DescribeRunner
from runners.runners import NMNRunner

def get_args():

	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('selection', choices=['find', 'describe', 'measure', 'encoder', 'nmn'])
	parser.add_argument('--max-epochs', type=int,
		help='Max. training epochs')
	parser.add_argument('--batch-size', type=int)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--save', action='store_true',
		help='Save the module after every epoch.')
	parser.add_argument('--suffix', type=str,
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--learning-rate', type=float,
		help='Specify learning rate')
	parser.add_argument('--weight-decay', type=float, help='Weight decay')
	parser.add_argument('--dropout', action='store_true')
	parser.add_argument('--competition', choices=['post', 'pre', 'softmax'],
		help='(find) Activation competition: pre/post sigmoidal or ReLU+softmax.')
	parser.add_argument('--visualize', type=int,
		help='(find) Visualize a masking example every N%. 0 is disabled.')
	parser.add_argument('--validate', action='store_true',
		help='Run validation every 1% of the dataset')
	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()

	if args.selection == 'find':
		assert not args.validate, "Can't validate Find module"
	else:
		assert args.visualize is None,\
			"Only find module is subject to visualization."
		assert args.competition is None,\
			"Competition only applies to find module."

	kwargs = { k: v for k, v in vars(args).items() if v is not None }
	del kwargs['selection']

	runner = dict(
		encoder  = EncoderRunner,
		find     = FindRunner,
		measure  = MeasureRunner,
		describe = DescribeRunner,
		nmn      = NMNRunner
	)[args.selection](**kwargs)

	runner.run()
