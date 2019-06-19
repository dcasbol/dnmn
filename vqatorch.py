import json
import torch
import random
import numpy as np
from misc.constants import *
from misc.indices import QUESTION_INDEX, DESC_INDEX, FIND_INDEX, ANSWER_INDEX, UNK_ID
from torch.utils.data import Dataset
from misc.util import flatten
from misc.parse import parse_tree


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
		return "find", FIND_INDEX[parse] or UNK_ID
	head = parse[0]
	below = [ parse_to_layout(c) for c in parse[1:] ]
	modules_below, indices_below = _ziplist(*below)
	module_head = 'and' if head == 'and' else 'describe'
	index_head = DESC_INDEX[head] or UNK_ID
	modules_here = [module_head] + modules_below
	indices_here = [index_head] + indices_below
	return modules_here, indices_here

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
			features = list(zdata.values())[0]
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
			parses = [ parse_tree(p) for p in parse_strs ]
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
			layouts_names = flatten(layouts_names)
			layouts_indices = flatten(layouts_indices)

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

	def __init__(self, *args, filter_data=True, metadata=False, **kwargs):
		superobj = super(VQAFindDataset, self).__init__(*args, **kwargs)
		self._metadata = metadata
		"""
		if filter_data:
			neg_set = {ANSWER_INDEX['no'], ANSWER_INDEX['0']}
			self._imap = list()
			for i, qid in enumerate(self._id_list):
				q = self._by_id[qid]
				if len(q['layouts_names']) != 2:
					continue
				head = q['parses'][0][0]
				if head in {'is', 'how_many'}:
					ans = set()
					for a in q['answers']:
						ans.add(a)
					if len(ans.intersection(neg_set)) > 0:
						continue
				self._imap.append(i)
		else:
			self._imap = list(range(len(self._id_list)))
		"""

		# THINK WELL IF WE MOVE TO THIS OR KEEP THE ABOVE
		neg_set = {ANSWER_INDEX['no'], ANSWER_INDEX['0']}
		self._imap = list()
		self._tmap = list()
		for i, qid in enumerate(self._id_list):
			q = self._by_id[qid]
			lnames = q['layouts_names']
			lindex = q['layouts_indices']
			head = q['parses'][0][0]
			if filter_data and head in {'is', 'how_many'}:
				ans = { a for a in q['answers'] }
				if len(ans.intersection(neg_set)) > 0:
					continue
			for j, (name, idx) in enumerate(zip(lnames, lindex)):
				if name != 'find':
					continue
				self._imap.append(i)
				self._tmap.append(j)

	def __len__(self):
		return len(self._imap)

	def __getitem__(self, i):
		datum, features = super(VQAFindDataset, self).__getitem__(self._imap[i])

		assert len(datum['parses']) == 1, 'Encountered item ({}) with +1 parses: {}'.format(i, datum['parses'])
		target = datum['layouts_indices']
		target = target[self._tmap[i]] if len(self._tmap) > 0 else target[-1]
		target_str = FIND_INDEX.get(target)
		
		output = (features, target)
		if self._metadata:
			output += (target_str, datum['input_set'], datum['input_id'])

		return output