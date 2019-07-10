import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX, QUESTION_INDEX, DESC_INDEX
from misc.constants import *
from misc.util import to_numpy


class InstanceModule(nn.Module):

	def __init__(self):
		super(InstanceModule, self).__init__()
		self._instance = None

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

	def __init__(self, competition='pre'):
		super(Find, self).__init__()
		assert competition in {'pre', 'post', None}, "Invalid competition mode: %s" % competition
		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)
		self._competition = competition

	def forward(self, features):
		c = self._get_instance()
		B = c.size(0)
		if self.training and self._competition is not None:
			if self._competition == 'post':
				mask_all = torch.sigmoid(self._conv(features))
				mask = mask_all[torch.arange(B), c].unsqueeze(1)
				mask_against = (mask_all.sum(1, keepdim=True) - mask) / (B-1)
				mask_train = mask / (1. + mask_against)
				mask_train = mask_train.view(B,-1).mean(1)
				return mask_train, mask
			else:
				# This one should work better
				h_all = self._conv(features)
				h = h_all[torch.arange(B), c].unsqueeze(1)
				h_against = (h_all.sum(1, keepdim=True) - h) / (B-1)
				h_against = F.relu(h_against)
				h_train = (h-h_against).view(B,-1).mean(1)
				mask = torch.sigmoid(h)
				self._instance = None
				return h_train, mask
		else:
			assert features.size(0) == B, 'features & c must have same size at dim 0'
			ks = self._conv.weight[c].unsqueeze(1).unbind(0)
			fs = features.unsqueeze(1).unbind(0)
			maps = torch.cat([ F.conv2d(f, k) for f, k in zip(fs, ks) ])
			masks = torch.sigmoid(maps)
			return masks


class Describe(InstanceModule):
	""" From 1st NMN article: It first computes an average over image features
	weighted by the attention, then passes this averaged feature vector through
	a single fully-connected layer. """

	def __init__(self):
		super(Describe, self).__init__()
		self._descr = list()
		for i in range(len(DESC_INDEX)):
			layer = nn.Linear(IMG_DEPTH, len(ANSWER_INDEX))
			setattr(self, '_descr_%d' % i, layer)
			self._descr.append(layer)

	def forward(self, mask, features):
		B,C,H,W = features.size()

		# Attend
		feat_flat = features.view(B,C,-1)
		mask_norm = mask.view(B,1,-1)
		attended = (mask_norm*feat_flat).mean(2)

		# Describe
		attended = attended.unsqueeze(1).unbind(0)
		instance = to_numpy(self._get_instance())
		preds = list()
		for att, inst in zip(attended, instance):
			preds.append(self._descr[inst](att))

		return torch.cat(preds)


class Measure(InstanceModule):

	def __init__(self):
		super(Measure, self).__init__()
		self._measure = list()
		for i in range(len(DESC_INDEX)):
			layers =  nn.Sequential(
				nn.Linear(MASK_WIDTH**2, HIDDEN_UNITS),
				nn.ReLU(),
				nn.Linear(HIDDEN_UNITS, len(ANSWER_INDEX))
			)
			setattr(self, '_measure_%d' % i, layers)
			self._measure.append(layers)

	def forward(self, mask):
		B = mask.size(0)
		mask = mask.view(B, -1).unsqueeze(1).unbind(0)
		instance = to_numpy(self._get_instance())
		preds = list()
		for m, inst in zip(mask, instance):
			preds.append(self._measure[inst](m))
		return torch.cat(preds)


class QuestionEncoder(nn.Module):

	def __init__(self):
		super(QuestionEncoder, self).__init__()
		self._wemb = nn.Embedding(len(QUESTION_INDEX), HIDDEN_UNITS)
		self._lstm = nn.LSTM(HIDDEN_UNITS, HIDDEN_UNITS)
		self._final = nn.Linear(HIDDEN_UNITS, len(ANSWER_INDEX))

	def forward(self, question, length):
		B = length.size(0)
		embed = self._wemb(question)
		hidden = self._lstm(embed)[0][length-1, torch.arange(B)]
		return self._final(hidden)
