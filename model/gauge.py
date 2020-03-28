from .base_module import BaseModule
from .find import Find
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.constants import *
from misc.indices import ANSWER_INDEX
from misc.util import generate_hmaps, attend_features


class GaugeFind(BaseModule):
	"""This module integrates a Gauge sub-module, which acts as an utility function
	for training the Find module"""

	NAME = 'gauge-find'

	def __init__(self, dropout=0, modular=False, softmax_attn=False, **kwargs):
		super(GaugeFind, self).__init__(dropout=dropout)
		self._classifier = nn.Sequential(
			nn.Linear(IMG_DEPTH + MASK_WIDTH**2, 64, bias=False),
			nn.Linear(64, len(ANSWER_INDEX))
		)
		self._find = Find(modular=modular, **kwargs)
		self._forced_dropout = lambda x: F.dropout(x, p=0.3, training=True)
		self._modular = modular
		self._softmax_attn = softmax_attn

	def forward(self, features, inst_1, inst_2, yesno, prior=None):

		features = self._dropout2d(features)

		instances = [inst_1, inst_2] if (inst_2>0).any() else [inst_1]
		hmap = generate_hmaps(self._find, instances, features, self._modular)

		B = hmap.size(0)
		yesno = yesno.view(B,1).float()
		attended  = attend_features(features, hmap, softmax=self._softmax_attn)
		hmap_flat = hmap.view(B,-1)
		x = torch.cat([(1.-yesno)*attended, yesno*hmap_flat], 1)
		if self.training:
			pred = self._classifier(self._forced_dropout(x))
			if prior is not None:
				pred = pred + prior
			return pred
		else:
			x = x.view(B,1,-1).expand(-1,20,-1)
			preds = self._classifier(self._forced_dropout(x))
			if prior is not None:
				preds = preds + prior.unsqueeze(1)
			preds = preds.softmax(dim=2)
			mean = preds.mean(1)
			idx  = mean.argmax(1)
			var  = preds[range(B),:,idx].var(1)
			return mean, var

	def save(self, filename):
		self._find.save(filename)

	def load(self, filename):
		self._find.load(filename)
