from .vqa_ds import VQADataset
from misc.util import is_yesno
from misc.indices import NULL_ID
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


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
