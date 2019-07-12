import torch
import numpy as np
from collections import defaultdict
from misc.indices import YESNO_QWORDS, OR_QWORD
from time import time


USE_CUDA = torch.cuda.is_available()
def cudalize(*x):
	x = [ xi.cuda() for xi in x ] if USE_CUDA else x
	return x[0] if len(x) == 1 else x

def to_numpy(x):
	return x.detach().cpu().numpy()

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

def top1_accuracy(pred, label):
	return (pred.argmax(1) == label).float().mean().item()

def inset_accuracy(pred, label_dist):
	idx = torch.arange(pred.size(0))
	hit = label_dist[idx, pred.argmax(1)] > 0
	return hit.float().mean().item()

def weighted_accuracy(pred, label_dist):
	idx = torch.arange(pred.size(0))
	return label_dist[idx, pred.argmax(1)].mean().item()

def is_yesno(q):
	return q[1] in YESNO_QWORDS and OR_QWORD not in q

def lookahead(iterable):
	it = iter(iterable)
	last = next(it)
	for val in it:
		yield last, False
		last = val
	yield last, True


class Chronometer(object):

	def __init__(self):
		self._t = 0.
		self._t0 = 0.
		self._running = False

	def read(self):
		return self._t + time() - self._t0 if self._running else self._t

	def reset(self):
		self._t = 0.
		self._t0 = time()

	def start(self):
		assert not self._running, "Chronometer already started"
		self._t0 = time()
		self._running = True

	def stop(self):
		assert self._running, "Chronometer already stopped"
		self._t += time() - self._t0
		self._running = False