import os
import json
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from misc.constants import *
from misc.indices import QUESTION_INDEX, ANSWER_INDEX, UNK_ID
from misc.util import flatten, ziplist
from misc.parse import parse_tree, process_question, parse_to_layout
from collections import defaultdict

class VQADataset(Dataset):
	"""
	VQA Dataset is composed from images and questions related to those.
	Each question points to an image. From each question (parse) at least
	one layout is extracted.
	"""

	def __init__(self, root_dir='./', set_names='train2014',
		start=None, stop=None, k=None, partition=None):
		super(VQADataset, self).__init__()
		self._root_dir = os.path.expanduser(root_dir)
		if type(set_names) == str:
			set_names = [set_names]
		for name in set_names:
			assert name in {'train2014', 'val2014', 'test2015'}, '{!r} is not a valid set'.format(name)
		self._set_names = set_names

		self._load_from_cache(set_names)
		self._id_list = list(self._by_id.keys())
		self._id_list.sort() # Ensure same order in all systems
		random.Random(0).shuffle(self._id_list)

		if k is not None:
			assert start is None and stop is None
			self._kfoldselection(k, partition)
		if start is not None:
			start = int(start*len(self._id_list))
		if stop is not None:
			stop = int(stop*len(self._id_list))
		if start is not None or stop is not None:
			self._id_list = self._id_list[slice(start, stop)]
		print('{} samples in dataset.'.format(len(self._id_list)))

		with np.load(NORMALIZERS_FILE) as zdata:
			self._mean = zdata['mean'].astype(np.float32)
			self._std  = zdata['std'].astype(np.float32)

	def _get_datum(self, i):
		return self._by_id[self._id_list[i]]

	def _get_features(self, datum):
		input_set, input_id = datum['input_set'], datum['input_id']
		input_path = IMAGE_FILE % (input_set, input_set, input_id)
		try:
			with open(input_path, 'rb') as fd:
				features = pickle.load(fd).toarray()
				features.shape = (MASK_WIDTH, MASK_WIDTH, IMG_DEPTH)
		except Exception as e:
			print('Error while loading features.')
			print('input_path:', input_path)
			raise e
		#features = (features - self._mean) / self._std
		# Positive values make more sense for a conv without bias
		features = features / (2*self._std)
		return features.transpose([2,0,1])

	def __len__(self):
		return len(self._id_list)

	def __getitem__(self, i):
		return self._get_datum(i)

	def _kfoldselection(self, k, partition):
		assert k in range(5)
		assert partition in ['train','val','test']
		assert set(self._set_names) == {'train2014','val2014'}

		# Group by input_id (image). Every image has 3 questions.
		by_input_id = defaultdict(list)
		for i, qid in enumerate(self._id_list):
			datum = self._get_datum(i)
			by_input_id[datum['input_id']].append(qid)

		input_id_list = list(by_input_id.keys())
		input_id_list.sort()
		random.Random(0).shuffle(input_id_list)

		fold_size = len(input_id_list) / 5
		limits = [0] + [ int((i+1)*fold_size) for i in range(5) ]
		folds  = [ input_id_list[limits[i]:limits[i+1]] for i in range(5) ]
		print([ len(f) for f in folds ])
		if partition == 'test':
			input_id_list = folds[(k-1)%5]
		else:
			input_id_list = list()
			for i in range(4):
				input_id_list.extend(folds[(k+i)%5])
			val_size = int(0.1*len(input_id_list))
			if partition == 'train':
				input_id_list = input_id_list[:-val_size]
			else:
				input_id_list = input_id_list[-val_size:]

		input_id_list = set(input_id_list)
		self._id_list = [ k for k, d in self._by_id.items() if d['input_id'] in input_id_list ]

	def _load_from_cache(self, set_names):
		import os
		import pickle
		sets_str = '_'.join(set_names)
		cache_fn = 'cache/{}.dat'.format(sets_str)
		cache_fn = os.path.join(self._root_dir, cache_fn)
		if os.path.exists(cache_fn):
			print('Loading from cache file %s' % cache_fn)
			with open(cache_fn, 'rb') as fd:
				self._by_id = pickle.load(fd)
		else:
			self._by_id = dict()
			for set_name in set_names:
				self._load_questions(set_name)
			print('Saving to cache file %s' % cache_fn)
			with open(cache_fn, 'wb') as fd:
				pickle.dump(self._by_id, fd, protocol=pickle.HIGHEST_PROTOCOL)

	def _load_questions(self, set_name):
		print('Loading questions from set %s' % set_name)
		question_fn = os.path.join(self._root_dir, QUESTION_FILE % set_name)
		parse_fn = os.path.join(self._root_dir, MULTI_PARSE_FILE % set_name)
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

			# CVPR2016 implementation. Don't ask.
			if parses[0][0] == "is":
				parses = parses[-1:]
			else:
				parses = parses[:1]

			layouts = [ parse_to_layout(p) for p in parses ]
			
			layouts_names, layouts_indices = ziplist(*layouts)
			layouts_names = flatten(layouts_names)
			layouts_indices = flatten(layouts_indices)

			question_id = question['question_id']
			datum = dict(
				question_id = question_id,
				question = indexed_question,
				parses = parses,
				layouts_names = layouts_names,
				layouts_indices = layouts_indices,
				input_set = set_name,
				input_id = question["image_id"],
				answers = []
			)
			self._by_id[question_id] = datum

		if set_name != "test2015":
			ann_fn = os.path.join(self._root_dir, ANN_FILE % set_name)
			with open(ann_fn) as ann_f:
				annotations = json.load(ann_f)["annotations"]
			for ann in annotations:
				question_id = ann["question_id"]
				if question_id not in self._by_id:
					continue

				indexed_answers = [ ANSWER_INDEX[a['answer']] or UNK_ID for a in ann['answers'] ]
				self._by_id[question_id]['answers'] = indexed_answers