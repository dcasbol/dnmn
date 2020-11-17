import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
from misc.util import program_depth
from misc.indices import Index, UNK, _index_words
from collections import defaultdict
from misc.constants import *

class CLEVRDataset(Dataset):

	def __init__(self, json_path=None, min_prog_depth=None, max_prog_depth=None,
		answer_index=None, find_index=None, desc_index=None, rel_index=None,
		clevr_dir='/DataSets/CLEVR_v1.0/'):

		self._clevr_dir = clevr_dir
		self._feats_pat = os.path.join(clevr_dir, 'conv/{split}/{name}.csr')
		if json_path is None:
			json_path = os.path.join(clevr_dir, 'questions/CLEVR_train_questions.json')

		spatial = np.mgrid[0:15,0:10].astype(np.float32)
		spatial[0] /= 15
		spatial[1] /= 10
		self._spatial = spatial
		with open('misc/clevr_normalizers.dat','rb') as fd:
			normalizers = pickle.load(fd)
		self._mean = normalizers['mean']
		self._std  = normalizers['std']

		print('Loading question data')
		with open(json_path) as fd:
			data = json.load(fd)['questions']

		self._questions = data
		if min_prog_depth is not None or max_prog_depth is not None:

			min_idx = '' if min_prog_depth is None else min_prog_depth
			max_idx = '' if max_prog_depth is None else max_prog_depth+1
			prog_range = '[{}:{}]'.format(min_idx, max_idx)
			print('Filtering by program depth in', prog_range)

			min_idx = min_prog_depth or 0
			max_idx = max_prog_depth or 100
			self._questions = list()
			for q in data:
				prog  = q['program']
				depth = program_depth(prog)
				if depth >= min_prog_depth and depth <= max_prog_depth:
					self._questions.append(q)

		if answer_index is None:
			print('Building answer index')

			answer_index = Index()
			answer_index.index(UNK)
			answer_counts = defaultdict(lambda: 0)
			for q in self._questions:
				answer_counts[q['answer']] += 1

			keep_answers = sorted([(c, a) for a, c in answer_counts.items()], reverse=True)
			keep_answers = list(keep_answers)[:MAX_ANSWERS]
			for count, answer in keep_answers:
				answer_index.index(answer)

		self._answer_index = answer_index

		def find_rule(find_counts, instr):
			if 'filter' in instr['function']:
				find_counts[instr['value_inputs'][0]] += 1

		def desc_rule(desc_counts, instr):
			if 'query' in instr['function']:
				attr = instr['function'].split('_')[1]
				desc_counts[attr] += 1

		def rel_rule(rel_counts, instr):
			funattr = instr['function'].split('_')
			if funattr[0] in ['same', 'relate']:
				rel_counts[funattr[1]] += 1

		indices = list()
		for name, (index, rule) in {
			'find': (find_index, find_rule),
			'desc': (desc_index, desc_rule),
			'rel' : (rel_index, rel_rule)
		}.items():

			if index is not None:
				indices.append(index)
				continue

			print('Building {} index'.format(name))

			index = Index()
			index.index(UNK)
			if name == 'find':
				index.index('*all*')
			counter = defaultdict(lambda: 0)
			for q in self._questions:
				for instr in q['program']:
					rule(counter, instr)

			threshold = 100
			_index_words(index, counter, threshold)
			indices.append(index)

		self._find_index, self._desc_index, self._rel_index = indices
		self._UNK_IDX = self._find_index[UNK]

		print('Dataset ready')

	def __len__(self):
		return len(self._questions)

	def __getitem__(self, i):

		q = self._questions[i]

		program = list()
		for instruction in q['program']:
			function = instruction['function']
			f, attr = function.split('_') if '_' in function else function, None

			if f == 'scene':
				instr = dict(
					module = 'find',
					instance = self._find_index['*all*']
				)
			elif f in ['exist','count']:
				instr = dict(
					module = 'measure',
					instance = ['exist','count'].index(f)
				)
			elif f in ['equal','less','greater']:
				instr = dict(
					module = 'compare',
					instance = ['equal','less','greater'].index(f)
				)
			elif f == 'filter':
				instance = self._find_index[instruction['value_inputs'][0]]
				instr = dict(
					module = 'find',
					instance = instance or self._UNK_IDX
				)
			elif f == 'intersect':
				instr = dict(module = 'and')
			elif f == 'query':
				instr = dict(
					module = 'describe',
					instance = self._desc_index[attr] or self._UNK_IDX
				)
			elif f == 'relate':
				instance = self._rel_index[instruction['value_inputs'][0]]
				instr = dict(
					module = 'relate',
					instance = instance or self._UNK_IDX
				)
			elif f == 'same':
				instr = dict(
					module = 'relate',
					instance = self._rel_index[attr] or self._UNK_IDX
				)
			elif f == 'union':
				instr = dict(module = 'or')
			elif f == 'unique':
				instr = dict(module = 'attention')
			else:
				raise ValueError('Unknown CLEVR function {}'.format(function))

			instr['inputs'] = instruction['inputs']
			program.append(instr)

		feats_file = self._feats_pat.format(
			split = q['split'],
			name  = q['image_filename'][:-4]
		)

		try:
			with open(feats_file, 'rb') as fd:
				features = pickle.load(fd).toarray()
		except Exception as e:
			print('Error while loading features.')
			print('from path:', feats_file)

		features.shape = (10,15,512)
		features = features / self._std
		features = features.transpose([2,0,1])
		features = np.concatenate([features, self._spatial], axis=0)

		answer = self._answer_index[q['answer']] or self._UNK_IDX

		return dict(
			program  = program,
			features = features,
			answer   = answer
		)