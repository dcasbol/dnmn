import torch
import torch.nn as nn
from .instance_module import InstanceModule
from misc.constants import *
from misc.indices import ANSWER_INDEX, DESC_INDEX


class Measure(InstanceModule):
	""" measure[c] takes an attention alone and maps it to a distribution over labels.
	hmap -> FC -> ReLU -> FC -> Softmax -> ans """

	NAME = 'measure'

	def __init__(self, hidden_size=HIDDEN_SIZE, hidden_dropout=0, **kwargs):
		super(Measure, self).__init__(**kwargs)
		self._measure = nn.ModuleList([
			nn.Sequential(
				nn.Linear(MASK_WIDTH**2, hidden_size),
				# Keep it as single-layer to preserve state_dict naming
				nn.Sequential(nn.ReLU(), nn.Dropout(p=hidden_dropout)),
				nn.Linear(hidden_size, len(ANSWER_INDEX))
			)
			for _ in range(len(DESC_INDEX))
		])

	def forward(self, mask, *features_dummy, prior=None):
		B = mask.size(0)
		mask = self._dropout(mask)
		mask = mask.view(B, -1).unsqueeze(1).unbind(0)
		instance = self._get_instance()
		preds = list()
		for m, inst in zip(mask, instance):
			preds.append(self._measure[inst](m))
		preds = torch.cat(preds)
		if prior is not None:
			preds = preds + prior
		return preds