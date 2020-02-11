from .vqa_ds import VQADataset
import torch
from torch.utils.data._utils.collate import default_collate
from misc.indices import NULL_ID


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