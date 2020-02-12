import torch.nn as nn
from .instance_module import InstanceModule
from misc.indices import ANSWER_INDEX, DESC_INDEX
from misc.constants import *
from misc.util import attend_features


class Describe(InstanceModule):
	"""First computes an average over image features weighted by the attention,
	then passes this averaged feature vector through a single fully-connected layer."""

	NAME = 'describe'

	def __init__(self, **kwargs):
		super(Describe, self).__init__(**kwargs)
		self._descr = nn.ModuleList([
			nn.Linear(IMG_DEPTH, len(ANSWER_INDEX))
			for _ in range(len(DESC_INDEX))
		])

	def forward(self, hmap_or_attended, features=None):
		if features is None:
			attended = hmap_or_attended
		else:
			attended = attend_features(features, hmap_or_attended)

		attended = self._dropout(attended)
		attended = attended.unsqueeze(1).unbind(0)
		instance = self._get_instance()
		preds = list()
		for att, inst in zip(attended, instance):
			preds.append(self._descr[inst](att))

		return torch.cat(preds)