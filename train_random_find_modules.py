import argparse
import json
from random import random
from runners.runners import FindRunner

def get_args():
	parser = argparse.ArgumentParser(description='Train N random Find modules')
	parser.add_argument('start', type=int)
	parser.add_argument('end', type=int)
	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()

	for i in range(args.start, args.end):
		kwargs = dict(
			max_epochs     = int(1+random()*9),
			batch_size     = int(16+random()*(512-16)),
			learning_rate  = 1e-5+random()*(0.1-1e-5),
			weight_decay   = random(),
			dropout        = random()*0.9,
			early_stopping = False
		)

		runner = FindRunner(**kwargs)
		runner.run()
		base_name = 'find-rnd-{}'.format(i)
		runner.save_model(base_name+'.pt')

		kwargs['top_1'] = runner.last_acc
		with open(base_name+'.json', 'w') as fd:
			json.dump(kwargs, fd)
