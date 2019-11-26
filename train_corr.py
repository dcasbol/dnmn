import os
import argparse
import json
import skopt
from runners.runners import EncoderRunner, FindRunner, MeasureRunner, DescribeRunner
from runners.runners import NMNRunner, DescribeRunnerUncached, MeasureRunnerUncached
import torch
torch.cuda.empty_cache()

def get_args():

	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('--corr-data', default='gauge_corr_data.json')
	parser.add_argument('--find-pt-dir', default='find-rnd')
	parser.add_argument('--skopt-res', default='hyperopt/nmn/nmn-res.gz')
	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()

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

	bs, lr, do, wd = skopt.load(args.skopt_res).x
	runner = NMNRunner(
		max_epochs    = 100,
		validate      = True,
		batch_size    = int(bs),
		learning_rate = lr,
		dropout       = do,
		weight_decay  = wd,
		find_pt       = pt_file
	)
	runner.run()

	d = read()
	d['nmn_accs'][idx] = runner.best_acc
	write(d)

	print('Best validation accuracy:', runner.best_acc)
	print('Achieved at epoch', runner.best_epoch)
