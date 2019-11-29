import os
import argparse
import json
from random import random
from runners.runners import FindRunner

def get_args():
	parser = argparse.ArgumentParser(description='Train N random Find modules')
	parser.add_argument('start', type=int)
	parser.add_argument('end', type=int)
	parser.add_argument('--target-dir', default='find-rnd')
	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()

	BASE_PAT  = os.path.join(args.target_dir,'find-rnd-{}')

	if not os.path.exists(args.target_dir):
		os.makedirs(args.target_dir)

	for i in range(args.start, args.end):

		base_name = BASE_PAT.format(i)
		assert not os.path.exists(base_name+'.pt'), "{!r} already exists".format(base_name)

		kwargs = dict(
			max_epochs     = int(1+random()*20),
			batch_size     = int(16+random()*(512-16)),
			learning_rate  = 10**(-5+random()*4),
			weight_decay   = 10**(-10+random()*9),
			dropout        = random()*0.9,
			early_stopping = False
		)

		runner = FindRunner(**kwargs)
		runner.run()
		runner.save_model(base_name+'.pt')

		kwargs['top_1'] = runner.last_acc
		kwargs['var']   = runner.last_var
		with open(base_name+'.json', 'w') as fd:
			json.dump(kwargs, fd)
