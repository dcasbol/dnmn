import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX, QUESTION_INDEX, DESC_INDEX, NULL_ID
from misc.constants import *
from misc.util import to_numpy


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

	def __init__(self, competition, **kwargs):
		super(Find, self).__init__(**kwargs)

		assert competition in {'pre', 'post', 'softmax', None},\
			"Invalid competition mode: {}".format(competition)

		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)
		self._competition = competition
		self._loss_func = {
			'pre'     : nn.BCEWithLogitsLoss,
			'post'    : nn.BCELoss,
			'softmax' : nn.CrossEntropyLoss,
			None      : lambda reduction: None
		}[competition](reduction = 'sum')
		self._loss = None

	def forward(self, features):
		c = self._get_instance()
		B = c.size(0)
		if self.training and self._competition is not None:
			B_idx = torch.arange(B)
			features = self._dropout(features)
			kernel = self._dropout(self._conv.weight)
			h_all = F.conv2d(features, kernel)
			if self._competition == 'temp':
				H,W = h_all.size()[-2:]
				hmap = h_all[B_idx, c].unsqueeze(1)
				min_val = hmap.view(B,-1).min(1)
				hmap -= min_val
				self._loss = 0.
				return hmap
			if self._competition == 'post':
				mask_all = torch.sigmoid(h_all)
				mask = mask_all[B_idx, c].unsqueeze(1)
				mask_against = (mask_all.sum(1, keepdim=True) - mask) / (B-1)
				mask_train = mask / (1. + mask_against)
				mask_train = mask_train.view(B,-1).mean(1)
				self._loss = self._loss_func(mask_train, torch.ones_like(mask_train))
				return mask
			elif self._competition == 'pre':
				h = h_all[B_idx, c].unsqueeze(1)
				mask = torch.sigmoid(h)
				h_against = (h_all.relu().sum(1, keepdim=True) - h.relu()) / (B-1)
				h_train = (h-h_against).view(B,-1).mean(1)
				self._loss = self._loss_func(h_train, torch.ones_like(h_train))
				return mask
			elif self._competition == 'softmax':
				h_all = h_all.relu()
				mask  = h_all[B_idx, c].unsqueeze(1)
				gap   = h_all.view(B,h_all.size(1),-1).mean(2)
				self._loss = self._loss_func(gap, c)
				return mask
		else:
			B,C,H,W = features.size()
			ks = self._dropout(self._conv.weight[c])
			fs = self._dropout(features.contiguous().view(1,B*C,H,W))
			masks = F.conv2d(fs, ks, groups=B).view(B,1,H,W)
			if self._competition in [None, 'softmax']:
				masks = masks.relu()
			else:
				masks = torch.sigmoid(masks)
			return masks

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

	def forward(self, mask, features):
		B,C,H,W = features.size()
		features = self._dropout(features)

		# Attend
		feat_flat = features.view(B,C,-1)
		mask = mask.view(B,1,-1) + 1e-10
		if self._norm:
			mask -= mask.min(2, keepdim=True).values - 1e-10
			mask /= mask.max(2, keepdim=True).values
		attended = (mask*feat_flat).sum(2) / mask.sum(2)
		attended = self._dropout(attended)

		# Describe
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

	def __init__(self, dropout=0, embed_size=None):
		super(QuestionEncoder, self).__init__()
		embed_size = embed_size or EMBEDDING_SIZE
		self._wemb = nn.Embedding(len(QUESTION_INDEX), embed_size,
			padding_idx=NULL_ID)
		self._lstm = nn.LSTM(embed_size, HIDDEN_UNITS)
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
