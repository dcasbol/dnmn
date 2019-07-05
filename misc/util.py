import torch
from collections import defaultdict

USE_CUDA = torch.cuda.is_available()
def cudalize(*x):
	x = [ xi.cuda() for xi in x ] if USE_CUDA else x
	return x[0] if len(x) == 1 else x

to_numpy = lambda x: x.detach().cpu().numpy()

def ziplist(*args):
	""" Original zip returns list of tuples """
	return [ [ a[i] for a in args] for i in range(len(args[0])) ]

def flatten(lol):
	if isinstance(lol, (tuple, list)):
		return sum([flatten(l) for l in lol], [])
	else:
		return [lol]

def postorder(tree):
	if isinstance(tree, tuple):
		for subtree in tree[1:]:
			for node in postorder(subtree):
				yield node
		yield tree[0]
	else:
		yield tree

def tree_map(function, tree):
	if isinstance(tree, tuple):
		head = function(tree)
		tail = tuple(tree_map(function, subtree) for subtree in tree[1:])
		return (head,) + tail
	return function(tree)

def tree_zip(*trees):
	if isinstance(trees[0], tuple):
		zipped_children = [[t[i] for t in trees] for i in range(len(trees[0]))]
		zipped_children_rec = [tree_zip(*z) for z in zipped_children]
		return tuple(zipped_children_rec)
	return trees

def max_divisor_batch_size(dataset_len, max_batch_size):
	batch_size = max_batch_size
	while dataset_len/float(batch_size) > dataset_len//batch_size:
		batch_size -= 1
	return batch_size

def majority_label(label_list):
	count = defaultdict(lambda: 0)
	for l in label_list:
		count[l] += 1
	label = max(count.keys(), key=lambda k: count[k])
	return label

def values_to_distribution(values, size):
	freqs = defaultdict(lambda: 0)
	for v in values:
		freqs[v] += 1
	total = sum(freqs.values())
	distr = np.zeros(size, dtype=np.float32)
	for v, f in freqs.items():
		distr[v] = f/total
	return distr

def accuracy(pred, label):
	return (pred.argmax(dim=1) == label).float().mean()
