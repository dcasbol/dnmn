import os
import json
import torch
import random
import numpy as np
from misc.constants import *
from misc.indices import QUESTION_INDEX, DESC_INDEX, FIND_INDEX, ANSWER_INDEX
from misc.indices import UNK_ID, NULL_ID, NEG_ANSWERS
from torch.utils.data import Dataset
from misc.util import flatten, ziplist, majority_label, values_to_distribution, is_yesno
from misc.parse import parse_tree, process_question, parse_to_layout
from functools import reduce


class VQADataset(Dataset):
	"""
	VQA Dataset is composed from images and questions related to those.
	Each question points to an image. From each question (parse) at least
	one layout is extracted.
	"""

	def __init__(self, root_dir='./', set_names='train2014', features=True):
		super(VQADataset, self).__init__()
		self._root_dir = os.path.expanduser(root_dir)
		if type(set_names) == str:
			set_names = [set_names]
		for name in set_names:
			assert name in {'train2014', 'val2014', 'test2015'}, '{!r} is not a valid set'.format(name)
		self._features = features
		self._set_names = set_names

		self._load_from_cache(set_names)
		self._id_list = list(self._by_id.keys())
		self._id_list.sort() # Ensure same order in all systems

		with np.load(NORMALIZERS_FILE) as zdata:
			self._mean = zdata['mean'].astype(np.float32)
			self._std  = zdata['std'].astype(np.float32)

	def _get_datum(self, i):
		return self._by_id[self._id_list[i]]

	def _get_features(self, datum):
		input_set, input_id = datum['input_set'], datum['input_id']
		input_path = IMAGE_FILE % (input_set, input_set, input_id)
		try:
			features = list(np.load(input_path).values())[0]
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
		# Get question data and load image features
		datum = self._get_datum(i)
		if not self._features:
			return datum
		return datum, self._get_features(datum)

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


class VQAFindDataset(VQADataset):

	def __init__(self, *args, filter_data=True, metadata=False, 
		filtering='majority', **kwargs):
		super(VQAFindDataset, self).__init__(*args, **kwargs)
		self._metadata = metadata

		assert filtering in ['majority', 'all', 'any'], "Invalid filtering mode {!r}".format(filtering)
		filter_op = dict(
			majority = lambda ans: majority_label(ans) in NEG_ANSWERS,
			all      = lambda ans: set(ans).issubset(NEG_ANSWERS),
			any      = lambda ans: len(set(ans).intersection(NEG_ANSWERS)) > 0
		)[filtering]

		self._imap = list()
		self._tmap = list()
		n_filtered = n_included = 0
		for i, qid in enumerate(self._id_list):

			q = self._by_id[qid]
			if filter_data and filter_op(q['answers']):
				n_filtered += 1
				continue

			lnames = q['layouts_names']
			lindex = q['layouts_indices']
			for j, (name, idx) in enumerate(zip(lnames, lindex)):
				if name != 'find': continue
				n_included += 1
				self._imap.append(i)
				self._tmap.append(j)

		print(n_filtered, 'filtered out,', n_included, 'included')

	def get(self, i, load_features=True):
		prev_features = self._features
		self._features = load_features
		datum = super(VQAFindDataset, self).__getitem__(self._imap[i])
		self._features = prev_features
		if load_features:
			datum, features = datum

		assert len(datum['parses']) == 1, 'Encountered item ({}) with +1 parses: {}'.format(i, datum['parses'])
		target = datum['layouts_indices']
		target = target[self._tmap[i]] if len(self._tmap) > 0 else target[-1]
		target_str = FIND_INDEX.get(target)
		
		output = (features, target) if load_features else (target,)
		if self._metadata:
			output += (target_str, datum['input_set'], datum['input_id'])

		return output

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


class VQARootModuleDataset(VQADataset):

	def __init__(self, *args, exclude=None, **kwargs):
		super(VQARootModuleDataset, self).__init__(*args, **kwargs)
		self._interdir = os.path.join(self._root_dir, INTER_HMAP_FILE)
		assert exclude in {None, 'yesno', 'others'}, "Invalid value for 'exclude': {}".format(exclude)

		if exclude is not None:
			yesno_questions = exclude == 'yesno'
			new_id_list = list()
			for qid in self._id_list:
				if is_yesno(self._by_id[qid]['question']) == yesno_questions:
					new_id_list.append(did)
			self._id_list = new_id_list
			assert len(self._id_list) > 0, "No samples were found with exclude = {!r}".format(exclude)

	def _compose_hmap(self, datum):
		names   = datum['layouts_names']
		indices = datum['layouts_indices']

		# Get hmaps
		hmap_list = list()
		for name, index in zip(names, indices):
			if name != 'find': continue

			fn = self._interdir.format(
				set = datum['input_set'],
				cat = FIND_INDEX.get(index),
				id  = datum['input_id']
			)
			hmap = list(np.load(fn).values())[0]
			hmap_list.append(hmap)

		# Compose them with ANDs
		mask = reduce(lambda x,y: x*y, hmap_list)
		return mask

	def __getitem__(self, i):

		datum = super(VQARootModuleDataset, self).__getitem__(i)
		if self._features:
			datum, features = datum

		mask = self._compose_hmap(datum)
		label = majority_label(datum['answers'])
		distr = values_to_distribution(datum['answers'], len(ANSWER_INDEX))
		instance = datum['layouts_indices'][0]

		if self._features:
			return mask, features, instance, label, distr
		return mask, instance, label, distr


class VQADescribeDataset(VQARootModuleDataset):
	def __init__(self, *args, **kwargs):
		super(VQADescribeDataset, self).__init__(*args, **kwargs,
			features=True, exclude='yesno')

class VQAMeasureDataset(VQARootModuleDataset):
	def __init__(self, *args, **kwargs):
		super(VQAMeasureDataset, self).__init__(*args, **kwargs,
			features=False, exclude='others')

class VQAEncoderDataset(VQADataset):

	def __init__(self, *args, **kwargs):
		super(VQAEncoderDataset, self).__init__(*args, **kwargs, features=False)

	def __getitem__(self, i):
		datum = self._by_id[self._id_list[i]]
		question = datum['question']
		label = majority_label(datum['answers'])
		distr = values_to_distribution(datum['answers'], len(ANSWER_INDEX))
		return question, len(question), label, distr

def encoder_collate_fn(data):
	questions, lengths, labels, distrs = zip(*data)
	T = max(lengths)
	padded = [ q + [NULL_ID]*(T-l) for q, l in zip(questions, lengths) ]
	to_tens = lambda x, t: torch.tensor(x,
		dtype=getattr(torch, t), requires_grad=False)
	questions = to_tens(padded, 'long').transpose(0,1)
	lengths   = to_tens(lengths, 'long')
	labels    = to_tens(labels, 'long')
	distrs    = to_tens(distrs, 'float')
	return questions, lengths, labels, distrs

class VQANMNDataset(VQADataset):

	def __init__(self, *args, answers=True, **kwargs):
		super(VQANMNDataset, self).__init__(*args, **kwargs, features=True)
		self._skip_answers = 'test2015' in self._set_names or not answers

	def __getitem__(self, i):
		datum, features = super(VQANMNDataset, self).__getitem__(i)

		names   = datum['layouts_names']
		indices = datum['layouts_indices']
		find_indices = [ i for n, i in zip(names, indices) if n == 'find' ]
		root_index = indices[0]

		q = datum['question']
		item = (q, len(q), is_yesno(q), features, root_index, find_indices)
		if self._skip_answers:
			return item + (datum['question_id'],)

		label = majority_label(datum['answers'])
		distr = values_to_distribution(datum['answers'], len(ANSWER_INDEX))

		return item + (label, distr)

def nmn_collate_fn(data):
	""" Custom collate function for NMN model. Pads questions, computes answer
	probability distribution and gives find-instance indices as tuples,
	because nr of calls is variable."""

	unzipped = list(zip(*data))
	has_labels = len(unzipped) == 8
	questions, lengths, yesno, features, root_idx, indices = unzipped[:6]
	if has_labels:
		label, distr = unzipped[-2:]
	else:
		qids = unzipped[-1]

	T = max(lengths)
	padded = [ q + [NULL_ID]*(T-l) for q, l in zip(questions, lengths) ]

	to_tens = lambda x, t: torch.tensor(x,
		dtype=getattr(torch, t), requires_grad=False)
	batch = dict(
		question  = to_tens(padded, 'long').transpose(0,1),
		length    = to_tens(lengths, 'long'),
		yesno     = to_tens(yesno, 'uint8'),
		features  = to_tens(features, 'float'),
		find_inst = indices,
		root_inst = to_tens(root_idx, 'long')
	)
	if has_labels:
		batch['label'] = to_tens(label, 'long')
		batch['distr'] = to_tens(distr, 'float')
	else:
		batch['question_id'] = qids

	return batch
