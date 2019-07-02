import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX, QUESTION_INDEX
from misc.constants import *


class FindModule(nn.Module):
	"""This module corresponds to the original 'attend' in the NMN paper."""

	def __init__(self, competition='pre'):
		super(FindModule, self).__init__()
		assert competition in {'pre', 'post'}, "Invalid competition mode: %s" % competition
		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)
		self._competition = competition

	def forward(self, features, c):

		if self.training:
			B = c.size(0)

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
				return h_train, mask
		else:
			k = self._conv.weight[c]
			x = torch.sigmoid(F.conv2d(features, k))
		return x


class DescribeModule(nn.Module):
	""" From 1st NMN article: It first computes an average over image features
	weighted by the attention, then passes this averaged feature vector through
	a single fully-connected layer. """

	def __init__(self):
		super(DescribeModule, self).__init__()
		self._final = nn.Linear(IMG_DEPTH, len(ANSWER_INDEX))

	def forward(self, mask, features):
		B,C,H,W = features.size()
		feat_flat = features.view(B,C,-1)
		mask_norm = mask.view(B,1,-1)
		mask_norm = F.softmax(mask_norm)
		#mask_norm = mask_norm / mask_norm.sum(2, keepdim=True)
		attended = (mask_norm*feat_flat).sum(2)
		return self._final(attended)


class MeasureModule(nn.Module):

	def __init__(self):
		super(MeasureModule, self).__init__()
		self._layers = nn.Sequential(
			nn.Linear(MASK_WIDTH**2, HIDDEN_UNITS),
			nn.ReLU(),
			nn.Linear(HIDDEN_UNITS, len(ANSWER_INDEX))
		)

	def forward(self, mask):
		B = mask.size(0)
		return self._layers(mask.view(B,-1))


class QuestionEncoder(nn.Module):

	def __init__(self):
		super(QuestionEncoder, self).__init__()
		self._wemb = nn.Embedding(len(QUESTION_INDEX), HIDDEN_UNITS)
		self._lstm = nn.LSTM(HIDDEN_UNITS)
		self._final = nn.Linear(HIDDEN_UNITS, len(ANSWER_INDEX))

	def forward(self, question):
		embed = self._wemb(question)
		hidden = self._lstm(embed)[1]
		return self._final(hidden)