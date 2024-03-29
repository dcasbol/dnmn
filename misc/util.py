import os
import torch
import json
import numpy as np
import time
import pickle
import subprocess
import random
from scipy.sparse import csr_matrix
from collections import defaultdict
from misc.indices import YESNO_QWORDS, OR_QWORD, UNK_ID
from misc.constants import *
from ilock import ILock

def seed(seed_value=0, fully_deterministic=True):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	if fully_deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark     = False

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
	y = pred.argmax(1).view(-1,1)
	hits = (y == label) & (y != UNK_ID)
	hits = hits.float().sum(1)/3.0
	hits = torch.min(hits, torch.ones_like(hits))
	return hits.mean().item()

def inset_accuracy(pred, label):
	y = pred.argmax(1).view(-1,1)
	hits = (y == label) & (y != UNK_ID)
	hits = hits.any(1).float()
	return hits.mean().item()

def rel_accuracy(pred, label):
	y = pred.argmax(1).view(-1,1)
	hits = (y == label) & (y != UNK_ID)
	hits = hits.float().sum(1) / 10.0
	return hits.mean().item()

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

def time_iter(iterable, chrono):
	it = iter(iterable)
	while True:
		chrono.start()
		try:
			x = next(it)
		except StopIteration as e:
			chrono.stop()
			raise e
		chrono.stop()
		yield x

def attend_features(features, hmap, flatten=True, softmax=False):
	if flatten:
		B,C,H,W  = features.size()
		features = features.view(B,C,-1)
		hmap     = hmap.view(B,1,-1)
	if softmax:
		return (hmap.softmax(2)*features).sum(2)
	return (hmap*features).sum(2) / (hmap.sum(2) + 1e-10)

def generate_hmaps(find, instances, features, modular=False):
	and_fn = torch.min if modular else lambda x,y: x*y
	hmaps  = find[instances[0]](features)
	for inst in instances[1:]:
		valid = inst>0
		hmaps_inst = find[inst[valid]](features[valid])
		new_hmaps  = hmaps.clone()
		new_hmaps[valid] = and_fn(hmaps[valid], hmaps_inst)
		hmaps = new_hmaps
	return hmaps

def program_depth(program):
	depths = list()
	for inst in program:
		d = 0
		if len(inst['inputs']) > 0:
			d = max([ depths[i] for i in inst['inputs'] ]) + 1
		depths.append(d)
	return depths[-1]

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
		self._t  = t0
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

class GPUScheduler(object):

	def __init__(self, script_name):
		self._lock_name   = 'gpu-scheduler'
		self._book_file   = '/tmp/gpu-book.json'
		self._script_name = script_name
		env = os.environ.copy()
		if 'CUDA_VISIBLE_DEVICES' in env:
			del env['CUDA_VISIBLE_DEVICES']
		command = 'import sys; import torch; sys.exit(torch.cuda.device_count())'
		self._num_gpus = subprocess.run(['python','-c',command], env=env).returncode
		print(self._num_gpus, 'GPUs found')
		assert self._num_gpus > 0, 'No GPUs were found in system'

	def __enter__(self):
		with ILock(self._lock_name):
			if not os.path.exists(self._book_file):
				status = { gpu_id : None for gpu_id in range(self._num_gpus) }
			else:
				with open(self._book_file) as fd:
					status = json.load(fd)
			booked_id = None
			for gpu_id, booked_script  in status.items():
				if booked_script is None:
					booked_id = gpu_id
					break
			assert booked_id is not None, "There is no free GPU"
			status[booked_id] = self._script_name
			with open(self._book_file,'w') as fd:
				json.dump(status, fd)
		self._booked_id = booked_id
		return booked_id

	def __exit__(self, type, value, traceback):
		with ILock(self._book_file):
			with open(self._book_file) as fd:
				status = json.load(fd)
			assert status[self._booked_id] == self._script_name,\
				'Failed to free GPU: an unexpected script name was encountered'
			status[self._booked_id] = None
			with open(self._book_file,'w') as fd:
				json.dump(status, fd)
