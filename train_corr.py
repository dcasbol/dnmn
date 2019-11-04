import argparse
import json
from runners.runners import EncoderRunner, FindRunner, MeasureRunner, DescribeRunner
from runners.runners import NMNRunner, DescribeRunnerUncached, MeasureRunnerUncached

def get_args():

	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('selection',
		choices=['find', 'describe', 'measure', 'encoder', 'nmn',
			'describe_uncached', 'measure_uncached'])
	parser.add_argument('--max-epochs', type=int,
		help='Max. training epochs')
	parser.add_argument('--batch-size', type=int)
	parser.add_argument('--restore-pt')
	parser.add_argument('--save', action='store_true',
		help='Save the model regularly.')
	parser.add_argument('--suffix',
		help='Add suffix to saved files. Useful when training +1 simultaneously.')
	parser.add_argument('--learning-rate', type=float,
		help='Specify learning rate')
	parser.add_argument('--weight-decay', type=float)
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--visualize', type=int,
		help='(find) Visualize an output heatmap every N%. 0 is disabled.')
	parser.add_argument('--validate', action='store_true',
		help='Run validation after every epoch')
	parser.add_argument('--find-pt')
	return parser.parse_args()


if __name__ == '__main__':

	with open('gauge_corr_data.json') as fd:
		d = json.load(fd)

	idx = 0
	for i in range(len(d['pt_files'])):
		if d['nmn_accs'][i] == -1:
			d['nmn_accs'][i] = -2
			with open('gauge_corr_data.json', 'w') as fd:
				json.dump(d, fd)
			idx = i
			pt_file = 'find-rnd/' + d['pt_files'][i]
			break

	args = get_args()

	if args.selection != 'find':
		assert args.visualize is None,\
			"Only find module is subject to visualization."

	if args.selection[-8:] == 'uncached':
		assert args.find_pt is not None, "You must specify find module for uncached training."

	kwargs = { k: v for k, v in vars(args).items() if v is not None }
	del kwargs['selection']
	kwargs['find_pt'] = pt_file

	runner = dict(
		encoder  = EncoderRunner,
		find     = FindRunner,
		measure  = MeasureRunner,
		describe = DescribeRunner,
		nmn      = NMNRunner,
		describe_uncached = DescribeRunnerUncached,
		measure_uncached  = MeasureRunnerUncached
	)[args.selection](**kwargs)

	with open('gauge_corr_data.json') as fd:
		d = json.load(fd)
	d['nmn_accs'][idx] = runner.best_acc
	with open('gauge_corr_data.json', 'w') as fd:
		json.dump(d, fd)

	runner.run()
	print('Best validation accuracy:', runner.best_acc)
	print('Achieved at epoch', runner.best_epoch)