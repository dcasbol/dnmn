import torch
import torch.nn as nn
from .instance_module import InstanceModule
from misc.indices import ANSWER_INDEX, DESC_INDEX
from misc.constants import *
from misc.util import attend_features


class Describe(InstanceModule):
	"""First computes an average over image features weighted by the attention,
	then passes this averaged feature vector through a single fully-connected layer."""

	NAME = 'describe'

	def __init__(self, softmax_attn=False, hidden_size=None, 
		hidden_dropout=0, **kwargs):
		super(Describe, self).__init__(**kwargs)
		self._descr = nn.ModuleList([
			self._make_inst(hidden_size, hidden_dropout)
			for _ in range(len(DESC_INDEX))
		])
		self._softmax_attn = softmax_attn

	def _make_inst(self, hidden_size, hidden_dropout):
		if hidden_size is None:
			return nn.Linear(IMG_DEPTH, len(ANSWER_INDEX))
		return nn.Sequential(
			nn.Linear(IMG_DEPTH, hidden_size),
			nn.ReLU(),
			nn.Dropout(p=hidden_dropout),
			nn.Linear(hidden_size, len(ANSWER_INDEX))
		)

	def forward(self, hmap_or_attended, features=None, prior=None):
		if features is None:
			attended = hmap_or_attended
		else:
			attended = attend_features(features, hmap_or_attended,
				softmax=self._softmax_attn)

		attended = self._dropout(attended)
		attended = attended.unsqueeze(1).unbind(0)
		instance = self._get_instance()
		preds = list()
		for att, inst in zip(attended, instance):
			preds.append(self._descr[inst](att))
		preds = torch.cat(preds)
		if prior is not None:
			preds = preds + prior
		return preds
