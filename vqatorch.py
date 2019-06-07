import json
import sexpdata
import torch
from misc.constants import *
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from torch.utils.data import Dataset
import numpy as np
import random

def _parse_tree(p):
	if "'" in p:
		p = "none"
	parsed = sexpdata.loads(p)
	extracted = _extract_parse(parsed)
	return extracted

def _extract_parse(p):
	if isinstance(p, sexpdata.Symbol):
		return p.value()
	elif isinstance(p, int):
		return str(p)
	elif isinstance(p, bool):
		return str(p).lower()
	elif isinstance(p, float):
		return str(p).lower()
	return tuple(_extract_parse(q) for q in p)

def process_question(question):
	qstr = question.lower().strip()
	if qstr[-1] == "?":
		qstr = qstr[:-1]
	words = qstr.split()
	words = ["<s>"] + words + ["</s>"]
	return words

def _ziplist(*args):
	""" Original zip returns list of tuples """
	return [ [ a[i] for a in args] for i in range(len(args[0])) ]

def parse_to_layout(parse):
	"""
	All leaves become find modules, all internal
	nodes become transform or combine modules dependent
	on their arity, and root nodes become describe
	or measure modules depending on the domain.
	"""
	if isinstance(parse, str):
		return "find", MODULE_INDEX[parse] or UNK_ID
	head = parse[0]
	below = [ parse_to_layout(c) for c in parse[1:] ]
	modules_below, indices_below = _ziplist(*below)
	module_head = 'and' if head == 'and' else 'describe'
	index_head = MODULE_INDEX[head] or UNK_ID
	modules_here = [module_head] + modules_below
	indices_here = [index_head] + indices_below
	return modules_here, indices_here

def _flatten(list_tree):
	if type(list_tree) not in (list, tuple):
		return [list_tree]
	elif len(list_tree) == 0:
		return list_tree
	return _flatten(list_tree[0]) + _flatten(list_tree[1:])

def _create_one_hot(values, size):
	freqs = { v:0 for v in set(values) }
	for v in values:
		freqs[v] += 1
	total = sum(freqs.values())
	binary = np.zeros(size, dtype=np.float32)
	for v, f in freqs.items():
		binary[v] = f/total
	return binary


class VQADataset(Dataset):
	"""
	VQA Dataset is composed from images and questions related to those.
	Each question points to an image. From each question (parse) at least
	one layout is extracted.
	"""

	def __init__(self, root_dir, set_names):
		super(VQADataset, self).__init__()
		self._root_dir = root_dir
		if type(set_names) == str:
			set_names = [set_names]

		self._load_from_cache(set_names)
		self._id_list = list(self._by_id.keys())

		with np.load(NORMALIZERS_FILE) as zdata:
			self._mean = zdata['mean'].astype(np.float32)
			self._std  = zdata['std'].astype(np.float32)

	def __len__(self):
		return len(self._by_id)

	def __getitem__(self, i):
		# Get question data and load image features
		datum = self._by_id[self._id_list[i]]
		input_set, input_id = datum['input_set'], datum['input_id']
		input_path = IMAGE_FILE % (input_set, input_set, input_id)
		with np.load(input_path) as zdata:
			features = list(zdata.values())[0].astype(np.float32)
			features = (features - self._mean) / self._std
		return datum, features.transpose([2,0,1])

	def _load_from_cache(self, set_names):
		import os
		import pickle
		sets_str = '_'.join(set_names)
		cache_filename = 'cache/{}_{}.dat'.format(sets_str, CHOOSER)
		if os.path.exists(cache_filename):
			print('Loading from cache file %s' % cache_filename)
			with open(cache_filename, 'rb') as fd:
				self._by_id = pickle.load(fd)
		else:
			self._by_id = dict()
			for set_name in set_names:
				self._load_questions(set_name)
			print('Saving to cache file %s' % cache_filename)
			with open(cache_filename, 'wb') as fd:
				pickle.dump(self._by_id, fd, protocol=pickle.HIGHEST_PROTOCOL)

	def _load_questions(self, set_name):
		print('Loading questions from set %s' % set_name)
		question_fn = QUESTION_FILE % set_name
		parse_fn = MULTI_PARSE_FILE % set_name
		with open(question_fn) as question_f, open(parse_fn) as parse_f:
			questions = json.load(question_f)['questions']
			parse_groups = [ l.strip() for l in parse_f ]
			assert len(questions) == len(parse_groups)

		pairs = zip(questions, parse_groups)
		for question, parse_group in pairs:

			question_str = process_question(question['question'])
			indexed_question = [ QUESTION_INDEX[w] or UNK_ID for w in question_str ]
			
			parse_strs = parse_group.split(';')
			parses = [ _parse_tree(p) for p in parse_strs ]
			parses = [ ('_what', '_thing') if p == 'none' else p for p in parses ]

			# TODO What is this?
			if CHOOSER == "null":
				parses = [("_what", "_thing")]
			elif CHOOSER == "cvpr":
				if parses[0][0] == "is":
					parses = parses[-1:]
				else:
					parses = parses[:1]
			elif CHOOSER == "naacl":
				pass
			else:
				assert False

			layouts = [ parse_to_layout(p) for p in parses ]
			layouts_names, layouts_indices = _ziplist(*layouts)
			layouts_names = _flatten(layouts_names)
			layouts_indices = _flatten(layouts_indices)

			image_set_name = "test2015" if set_name == "test-dev2015" else set_name
			question_id = question['question_id']
			datum = dict(
				question_id = question_id,
				question = indexed_question,
				parses = parses,
				layouts_names = layouts_names,
				layouts_indices = layouts_indices,
				input_set = image_set_name,
				input_id = question["image_id"],
				answers = []
			)
			self._by_id[question_id] = datum

		if set_name not in ("test2015", "test-dev2015"):
			with open(ANN_FILE % set_name) as ann_f:
				annotations = json.load(ann_f)["annotations"]
			for ann in annotations:
				question_id = ann["question_id"]
				if question_id not in self._by_id:
					continue

				indexed_answers = [ ANSWER_INDEX[a['answer']] or UNK_ID for a in ann['answers'] ]
				self._by_id[question_id]['answers'] = indexed_answers


class VQAFindDataset(VQADataset):

	def __init__(self, *args):
		superobj = super(VQAFindDataset, self).__init__(*args)
		self._imap = [ i for i, qid in enumerate(self._id_list) if len(self._by_id[qid]['layouts_names']) == 2 ]

	def __len__(self):
		return len(self._imap)

	def __getitem__(self, i):
		datum, features = super(VQAFindDataset, self).__getitem__(self._imap[i])
		target = random.choice(datum['parses'])[-1]
		target = MODULE_INDEX[target] or UNK_ID
		
		input_set, input_id = datum['input_set'], datum['input_id']
		raw_input_path = RAW_IMAGE_FILE % (input_set, input_set, input_id)
		return features, target, raw_input_path