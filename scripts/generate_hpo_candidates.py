import random
import json

random.seed(0)

candidates = [ dict(
	batch_size    = random.randint(16, 512),
	dropout       = random.uniform(0, 0.9),
	learning_rate = 10**random.uniform(-5, -1),
	weight_decay  = 10**random.uniform(-10, 0)
) for _ in range(50) ]

with open('hpo_candidates.json','w') as fd:
	json.dump(candidates, fd)
