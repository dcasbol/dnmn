from .vqa_ds import VQADataset
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from misc.indices import NULL_ID


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