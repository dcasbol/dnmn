import torch
import json
import numpy as np
import time
from collections import defaultdict
from misc.indices import YESNO_QWORDS, OR_QWORD


USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
def cudalize(*x):
	x = [ xi.cuda() for xi in x ] if USE_CUDA else x
	return x[0] if len(x) == 1 else x

def cudalize_dict(d, exclude=[]):
	return { k : cudalize(v) if k not in exclude else v for k, v in d.items() }

def to_numpy(x):
	return x.detach().cpu().numpy()

def to_tens(x, t, d='cpu'):
	return torch.tensor(x, dtype=getattr(torch, t), requires_grad=False, device=d)

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

def attend_features(features, hmap, flatten=True):
	if flatten:
		B,C,H,W  = features.size()
		features = features.view(B,C,-1)
		hmap     = hmap.view(B,1,-1)
	return (hmap*features).sum(2) / (hmap.sum(2) + 1e-10)

def generate_hmaps(find, instances, features):
	hmaps = find[instances[0]](features)
	for inst in instances[1:]:
		valid = inst>0
		hmaps_inst = find[inst[valid]](features[valid])
		hmaps[valid] = hmaps[valid] * hmaps_inst
	return hmaps

class Logger(object):

	def __init__(self):
		self._log = defaultdict(lambda: [])

	def log(self, **kwargs):
		for k, v in kwargs.items():
			self._log[k].append(v)

	def save(self, filename):
		with open(filename, 'w') as fd:
			json.dump(self._log, fd)

	def load(self, filename):
		with open(filename, 'r') as fd:
			prev_log = json.load(fd)
			for key, values in prev_log.items():
				self._log[key].extend(values)

	def print(self, exclude=[]):
		for key, values in self._log.items():
			if key in exclude: continue
			print(key, ':', values[-1])

	def last(self, key):
		return self._log[key][-1]


class Chronometer(object):

	def __init__(self):
		self._t = 0.
		self._t0 = 0.
		self._running = False

	def read(self):
		return self._t + time.time() - self._t0 if self._running else self._t

	def read_str(self):
		return time.strftime('%H:%M:%S', time.localtime(self.read()))

	def reset(self):
		self._t = 0.
		self._t0 = time.time()

	def start(self):
		assert not self._running, "Chronometer already started"
		self._t0 = time.time()
		self._running = True

	def stop(self):
		assert self._running, "Chronometer already stopped"
		self._t += time.time() - self._t0
		self._running = False

	def set_t0(self, t0):
		self._t0 = time.time()


class PercentageCounter(object):

	def __init__(self, batch_size, dataset_len):
		self._last_perc   = -1
		self._batch_size  = batch_size
		self._dataset_len = dataset_len
		print('PercentageCounter initialized with len={}'.format(dataset_len))

	def update(self, batch_idx):
		perc = (batch_idx*self._batch_size*100)//self._dataset_len
		if perc == self._last_perc:
			return False
		self._last_perc = perc
		return True

	def float(self):
		return self._last_perc/100

	def __repr__(self):
		return '{: 3d}%'.format(self._last_perc)
