import os
import json
import torch
import pickle
import numpy as np
from misc.constants import *
from misc.indices import QUESTION_INDEX, FIND_INDEX, ANSWER_INDEX
from misc.indices import UNK_ID, NULL_ID, NEG_ANSWERS
from torch.utils.data import Dataset
from misc.util import flatten, ziplist, majority_label, is_yesno
from misc.parse import parse_tree, process_question, parse_to_layout
from torch.utils.data._utils.collate import default_collate


class VQADataset(Dataset):
	"""
	VQA Dataset is composed from images and questions related to those.
	Each question points to an image. From each question (parse) at least
	one layout is extracted.
	"""

	def __init__(self, root_dir='./', set_names='train2014',
		start=None, stop=None):
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

	def __len__(self):
		return len(self._imap)

	def __getitem__(self, i):
		datum    = self._get_datum(self._imap[i])
		features = self._get_features(datum)

		assert len(datum['parses']) == 1, 'Encountered item ({}) with +1 parses: {}'.format(i, datum['parses'])
		target = datum['layouts_indices']
		target = target[self._tmap[i]] if len(self._tmap) > 0 else target[-1]
		
		output = (features, target)
		if self._metadata:
			target_str = FIND_INDEX.get(target)
			output += (target_str, datum['input_set'], datum['input_id'])

		return output


class VQARootModuleDataset(VQADataset):

	def __init__(self, *args, exclude=None, **kwargs):
		super(VQARootModuleDataset, self).__init__(*args, **kwargs)
		self._hmap_pat = os.path.join(self._root_dir, CACHE_HMAP_FILE)
		self._att_pat = os.path.join(self._root_dir, CACHE_ATT_FILE)
		assert exclude in {None, 'yesno', 'others'}, "Invalid value for 'exclude': {}".format(exclude)

		if exclude is not None:
			yesno_questions = exclude == 'yesno'
			new_id_list = list()
			for qid in self._id_list:
				if is_yesno(self._by_id[qid]['question']) == yesno_questions:
					new_id_list.append(qid)
			self._id_list = new_id_list
			print('Filtered dataset has {} samples'.format(len(self._id_list)))
			assert len(self._id_list) > 0, "No samples were found with exclude = {!r}".format(exclude)

	def _get_hmap(self, datum):
		hmap_fn = self._hmap_pat.format(
			set = datum['input_set'],
			qid = datum['question_id']
		)
		return np.load(hmap_fn)

	def _get_attended(self, datum):
		att_fn = self._att_pat.format(
			set = datum['input_set'],
			qid = datum['question_id']
		)
		return np.load(att_fn)

	def __getitem__(self, i):
		datum = self._get_datum(i)
		labels = np.array(datum['answers'])
		instance = datum['layouts_indices'][0]
		return instance, labels, datum


class VQADescribeDataset(VQARootModuleDataset):
	def __init__(self, *args, **kwargs):
		super(VQADescribeDataset, self).__init__(*args, **kwargs, exclude='yesno')

	def __getitem__(self, i):
		instance, labels, datum = super(VQADescribeDataset, self).__getitem__(i)
		att = self._get_attended(datum)
		return att, instance, labels

class VQAMeasureDataset(VQARootModuleDataset):
	def __init__(self, *args, **kwargs):
		super(VQAMeasureDataset, self).__init__(*args, **kwargs, exclude='others')

	def __getitem__(self, i):
		instance, labels, datum = super(VQAMeasureDataset, self).__getitem__(i)
		hmap = self._get_hmap(datum)
		return hmap, instance, labels

class VQAEncoderDataset(VQADataset):

	def __init__(self, *args, **kwargs):
		super(VQAEncoderDataset, self).__init__(*args, **kwargs)

	def __getitem__(self, i):
		datum    = self._get_datum(i)
		question = datum['question']
		labels   = np.array(datum['answers'])
		return question, len(question), labels

def encoder_collate_fn(data):
	questions, lengths, labels = zip(*data)
	max_len = max(lengths)
	padded = [ q + [NULL_ID]*(max_len-l) for q, l in zip(questions, lengths) ]
	questions = torch.tensor(padded, dtype=torch.long).transpose(0,1)
	lengths, labels = default_collate(tuple(zip(lengths, labels)))
	return questions, lengths, labels

class VQAGaugeFindDataset(VQADataset):

	def __init__(self, *args, metadata=False, **kwargs):
		super(VQAGaugeFindDataset, self).__init__(*args, **kwargs)
		self._metadata = metadata

	def __getitem__(self, i):
		datum    = self._get_datum(i)
		features = self._get_features(datum)

		target_list = list()
		for name, index in zip(datum['layouts_names'], datum['layouts_indices']):
			if name != 'find': continue
			target_list.append(index)

		n = len(target_list)
		if n == 2:
			target_1, target_2 = target_list
		else:
			target_1 = target_list[0]
			target_2 = 0

		yesno  = is_yesno(datum['question'])
		labels = np.array(datum['answers'])
		
		output = (features, target_1, target_2, yesno, labels)
		if self._metadata:
			target_str = FIND_INDEX.get(target_1)
			if target_2 > 0:
				target_str += ' AND ' + FIND_INDEX.get(target_2)
			output += (target_str, datum['input_set'], datum['input_id'])

		return output


class VQANMNDataset(VQADataset):

	def __init__(self, *args, answers=True, **kwargs):
		super(VQANMNDataset, self).__init__(*args, **kwargs)
		self._skip_answers = 'test2015' in self._set_names or not answers

	def __getitem__(self, i):
		datum    = self._get_datum(i)
		features = self._get_features(datum)

		find_indices = list()
		for name, index in zip(datum['layouts_names'], datum['layouts_indices']):
			if name != 'find': continue
			find_indices.append(index)
		root_index = datum['layouts_indices'][0]

		n_indices = len(find_indices)
		q = datum['question']
		sample = (q, len(q), is_yesno(q), features, root_index, find_indices, n_indices)
		if self._skip_answers:
			return sample + (datum['question_id'], True)

		labels = np.array(datum['answers'])

		return sample + (labels,)


def nmn_collate_fn(data):
	unzipped   = list(zip(*data))
	has_labels = len(unzipped) == 8
	questions, lengths, yesno, features, root_idx, indices, num_idx = unzipped[:7]
	if has_labels:
		labels = unzipped[-1]
	else:
		qids   = unzipped[-2]

	max_len   = max(lengths)
	questions = [ q + [NULL_ID]*(max_len-l) for q, l in zip(questions, lengths) ]
	questions = torch.tensor(questions, dtype=torch.long).transpose(0,1)

	max_num = max(num_idx)
	indices = [ idxs + [0]*(max_num-n) for idxs, n in zip(indices, num_idx) ]
	indices = torch.tensor(indices, dtype=torch.long).unbind(1)

	batch = (lengths, yesno, features, root_idx)
	if has_labels:
		batch += (labels,)
	batch = default_collate(tuple(zip(*batch)))

	batch_dict = dict(
		length    = batch[0],
		yesno     = batch[1],
		features  = batch[2],
		root_inst = batch[3],
		question  = questions,
		find_inst = indices,
	)
	if has_labels:
		batch_dict['label'] = batch[4]
	else:
		batch_dict['question_id'] = qids

	return batch_dict


class CacheDataset(VQADataset):

	def __init__(self, features, questions, *args, **kwargs):
		super(CacheDataset, self).__init__(*args, **kwargs)
		assert features or questions, "Select loading features and/or questions"
		self._features  = features
		self._questions = questions

	def __getitem(self, i):
		datum  = self._get_datum(i)
		sample = { 'question_id' : datum['question_id'] }

		if self._features:
			find_indices = list()
			for name, index in zip(datum['layouts_names'], datum['layouts_indices']):
				if name != 'find': continue
				find_indices.append(index)
			sample.update(dict(
				features     = self._get_features(datum),
				find_indices = find_indices
			))

		if self._questions:
			q = datum['question']
			sample.update(dict(question = q, question_len = len(q)))
		
		return sample


def cache_collate_fn(data):

	batch = { 'question_id' : [ d['question_id'] for d in data ] }

	if 'question' in data[0]:
		questions = list()
		lengths   = list()
		for d in data:
			q = d['question']
			questions.append(q)
			lengths.append(len(q))
		max_len   = max(lengths)
		questions = [ q + [NULL_ID]*(max_len-l) for q, l in zip(questions, lengths) ]
		questions = torch.tensor(questions, dtype=torch.long).transpose(0,1)
		lengths   = default_collate(lengths)
		batch.update(dict(
			question = questions,
			length   = lengths
		))

	if 'features' in data[0]:
		indices = list()
		num_idx = list()
		for d in data:
			idcs = d['find_indices']
			indices.append(idcs)
			num_idx.append(len(idcs))
		max_num = max(num_idx)
		indices = [ idcs + [0]*(max_num-n) for idcs, n in zip(indices, num_idx) ]
		indices = torch.tensor(indices, dtype=torch.long).unbind(1)
		features = default_collate([ d['features'] for d in data ])
		batch.update(dict(
			features  = features,
			find_inst = indices
		))

	return batch