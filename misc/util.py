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

	NORMAL  = 0
	EXCLUDE = 1
	INCLUDE = 2

	def __init__(self):
		self.reset()

	def __enter__(self):
		assert self._mode != self.NORMAL
		if self._mode == self.EXCLUDE:
			self._t_begin = time()
		else:
			self._te += time() - self._t_begin

	def __exit__(self, *args):
		if self._mode == self.EXCLUDE:
			self._te += time() - self._t_begin
		else:
			self._t_begin = time()
		self._mode = self._stack.pop()

	def read(self):
		t = self._t_begin if self._mode == self.EXCLUDE else time()
		return t - self._t0 - self._te

	def reset(self):
		self._t0 = time()
		self._te = 0.
		self._mode = self.NORMAL
		self._stack = list()

	def exclude(self):
		assert self._mode != self.EXCLUDE
		self._stack.append(self._mode)
		self._mode = self.EXCLUDE
		return self

	def include(self):
		assert self._mode == self.EXCLUDE
		self._stack.append(self._mode)
		self._mode = self.INCLUDE
		return self