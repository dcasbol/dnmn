import torch
from collections import defaultdict

USE_CUDA = torch.cuda.is_available()
def cudalize(*x):
	x = [ xi.cuda() for xi in x ] if USE_CUDA else x
	return x[0] if len(x) == 1 else x

to_numpy = lambda x: x.detach().cpu().numpy()

class Struct:
	def __init__(self, **entries):
		rec_entries = {}
		for k, v in entries.items():
			if isinstance(v, dict):
				rv = Struct(**v)
			elif isinstance(v, list):
				rv = []
				for item in v:
					if isinstance(item, dict):
						rv.append(Struct(**item))
					else:
						rv.append(item)
			else:
				rv = v
			rec_entries[k] = rv
		self.__dict__.update(rec_entries)

	def __str_helper(self, depth):
		lines = []
		for k, v in self.__dict__.items():
			if isinstance(v, Struct):
				v_str = v.__str_helper(depth + 1)
				lines.append("%s:\n%s" % (k, v_str))
			else:
				lines.append("%s: %r" % (k, v))
		indented_lines = ["    " * depth + l for l in lines]
		return "\n".join(indented_lines)

	def __str__(self):
		return "struct {\n%s\n}" % self.__str_helper(1)

	def __repr__(self):
		return "Struct(%r)" % self.__dict__

class Index:
	def __init__(self):
		self.contents = dict()
		self.ordered_contents = []
		self.reverse_contents = dict()

	def __getitem__(self, item):
		if item not in self.contents:
			return None
		return self.contents[item]

	def index(self, item):
		if item not in self.contents:
			idx = len(self.contents) + 1
			self.ordered_contents.append(item)
			self.contents[item] = idx
			self.reverse_contents[idx] = item
		idx = self[item]
		assert idx != 0
		return idx

	def get(self, idx):
		if idx == 0:
			return "*invalid*"
		return self.reverse_contents[idx]

	def __len__(self):
		return len(self.contents) + 1

	def __iter__(self):
		return iter(self.ordered_contents)

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
