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

	def __init__(self, competition='pre', mask_norm='auto'):
		super(Find, self).__init__()
		
		if mask_norm == 'auto':
			mask_norm = None if competition == 'softmax' else 'sigmoid'
				
		assert competition in {'pre', 'post', 'softmax', None},\
			"Invalid competition mode: {}".format(competition)
		assert mask_norm in {'sigmoid', None},\
			"Invalid normalization mode: {}".format(mask_norm)

		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)
		self._competition = competition
		self._normalize_fn = {
			'sigmoid'   : torch.sigmoid,
			None        : lambda x: x
		}[mask_norm]
		self._loss_func = {
			'pre'     : nn.BCEWithLogitsLoss,
			'post'    : nn.BCELoss,
			'softmax' : nn.CrossEntropyLoss,
			None      : lambda reduction: None
		}[competition](reduction = 'sum')
		self._loss = None

	def _cross_entropy(x, y):
		# L(x) = -y*log(x) -(1-y)*log(1-x)
		x = x[torch.arange(x.size(0)), y]
		return -((x+1e-10).log() + (1.-x+1e-10).log()).sum()

	def forward(self, features):
		c = self._get_instance()
		B = c.size(0)
		if self.training and self._competition is not None:
			B_idx = torch.arange(B)
			h_all = self._conv(features)
			if self._competition == 'post':
				mask_all = torch.sigmoid(h_all)
				mask = mask_all[B_idx, c].unsqueeze(1)
				mask_against = (mask_all.sum(1, keepdim=True) - mask) / (B-1)
				mask_train = mask / (1. + mask_against)
				mask_train = mask_train.view(B,-1).mean(1)
				self._loss = self._loss_func(mask_train, torch.ones_like(mask_train))
				return mask
			elif self._competition == 'pre':
				# This one should work better
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
			ks = self._conv.weight[c].unsqueeze(1).unbind(0)
			fs = features.unsqueeze(1).unbind(0)
			maps = torch.cat([ F.conv2d(f, k) for f, k in zip(fs, ks) ])
			masks = torch.sigmoid(maps)
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
