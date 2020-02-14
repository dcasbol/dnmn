import random
import json
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Generate N random configurations')
	parser.add_argument('N', type=int, default=50)
	parser.add_argument('--output', type=str, default='hpo_candidates.json')
	parser.add_argument('--seed', type=int, default=0)
	return parser.parse_args()

def main(args):
	assert args.N > 0

	random.seed(args.seed)

	candidates = [ dict(
		batch_size    = random.randint(16, 512),
		dropout       = random.uniform(0, 0.9),
		learning_rate = 10**random.uniform(-5, -1),
		weight_decay  = 10**random.uniform(-10, 0)
	) for _ in range(args.N) ]

	with open(args.output,'w') as fd:
		json.dump(candidates, fd)

if __name__ == '__main__':
	main(get_args())