import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX, QUESTION_INDEX, DESC_INDEX, NULL_ID
from misc.constants import *
from misc.util import to_numpy, attend_features


class InstanceModule(nn.Module):

	def __init__(self, dropout=0):
		super(InstanceModule, self).__init__()
		self._instance = None
		self._dropout = {
			False : lambda x: x,
			True  : lambda x: F.dropout(x, p=dropout, training=self.training)
		}[dropout>0]

	def __getitem__(self, instance):
		assert self._instance is None
		self._instance = instance
		return self

	def _get_instance(self):
		inst = self._instance
		assert inst is not None, "Can't call module without instance"
		self._instance = None
		return inst


class Find(InstanceModule):
	"""This module corresponds to the original 'attend' in the NMN paper."""

	NAME = 'find'

	def __init__(self, **kwargs):
		super(Find, self).__init__(**kwargs)
		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)
		self._conv.weight.data.fill_(0.5)
		self._loss = None

	def forward(self, features):
		c = self._get_instance()
		B,D,H,W = features.size()
		kernel = self._dropout(self._conv.weight[c])
		group_feats = features.contiguous().view(1,B*D,H,W)
		hmap = F.conv2d(group_feats, kernel, groups=B).relu()
		if self.training:
			hmap_flat = hmap.view(B,-1)
			max_val = hmap_flat.max(1, keepdim=True).values
			hmap_norm = hmap_flat / (max_val+1e-10)
			self._loss = -hmap_norm.mean(1).sum()
		return hmap.view(B,1,H,W)

	def loss(self):
		assert self._loss is not None, "Call to loss must be preceded by a forward call."
		loss = self._loss
		self._loss = None
		return loss

class Describe(InstanceModule):
	""" From 1st NMN article: It first computes an average over image features
	weighted by the attention, then passes this averaged feature vector through
	a single fully-connected layer. """
	NAME = 'describe'

	def __init__(self, normalize_attention=False, **kwargs):
		super(Describe, self).__init__(**kwargs)
		self._descr = list()
		for i in range(len(DESC_INDEX)):
			layer = nn.Linear(IMG_DEPTH, len(ANSWER_INDEX))
			setattr(self, '_descr_%d' % i, layer)
			self._descr.append(layer)
		self._norm = normalize_attention

	def forward(self, hmap_or_attended, features=None):
		if features is None:
			attended = attend_features(features, hmap_or_attended)
		else:
			attended = hmap_or_attended

		attended = self._dropout(attended)
		attended = attended.unsqueeze(1).unbind(0)
		instance = self._get_instance()
		preds = list()
		for att, inst in zip(attended, instance):
			preds.append(self._descr[inst](att))

		return torch.cat(preds)


class Measure(InstanceModule):

	NAME = 'measure'

	def __init__(self, **kwargs):
		super(Measure, self).__init__(**kwargs)
		self._measure = list()
		for i in range(len(DESC_INDEX)):
			layers =  nn.Sequential(
				nn.Linear(MASK_WIDTH**2, HIDDEN_SIZE),
				nn.ReLU(),
				nn.Linear(HIDDEN_SIZE, len(ANSWER_INDEX))
			)
			setattr(self, '_measure_%d' % i, layers)
			self._measure.append(layers)

	def forward(self, mask, *dummy_args):
		B = mask.size(0)
		mask = self._dropout(mask)
		mask = mask.view(B, -1).unsqueeze(1).unbind(0)
		instance = self._get_instance()
		preds = list()
		for m, inst in zip(mask, instance):
			preds.append(self._measure[inst](m))
		return torch.cat(preds)


class QuestionEncoder(nn.Module):

	NAME = 'encoder'

	def __init__(self, dropout=0):
		super(QuestionEncoder, self).__init__()
		self._wemb = nn.Embedding(len(QUESTION_INDEX), EMBEDDING_SIZE)
		self._lstm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_UNITS)
		self._final = nn.Linear(HIDDEN_UNITS, len(ANSWER_INDEX))
		self._dropout = {
			False : lambda x: x,
			True  : lambda x: F.dropout(x, p=dropout, training=self.training)
		}[dropout>0]

	def forward(self, question, length):
		B = length.size(0)
		embed = self._wemb(question)
		hidden = self._lstm(embed)[0][length-1, torch.arange(B)]
		return self._final(self._dropout(hidden))
