import random
import json
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Generate N random configurations')
	parser.add_argument('N', type=int, default=50)
	parser.add_argument('--output', type=str, default='hpo_candidates.json')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--selection',
		choices=['find', 'describe', 'measure', 'encoder', 'nmn'],
		default='nmn')
	return parser.parse_args()

def gen_candidate(sel):
	c = dict(
		batch_size    = random.randint(16, 512),
		dropout       = random.uniform(0, 0.9),
		learning_rate = 10**random.uniform(-5, -1),
		weight_decay  = 10**random.uniform(-10, 0)
	)
	if sel != 'nmn':
		c['batch_size'] = random.randint(16, 2048)
	if sel == 'encoder':
		c['embedding_size'] = random.randint(16, 1000)
		c['hidden_units']   = random.randint(16, 1024)
	return c

def main(args):
	assert args.N > 0

	random.seed(args.seed)

	candidates = [ gen_candidate(args.selection) for _ in range(args.N) ]

	with open(args.output,'w') as fd:
		json.dump(candidates, fd)

if __name__ == '__main__':
	main(get_args())