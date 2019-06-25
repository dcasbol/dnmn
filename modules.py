import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX
from misc.constants import *
from misc.util import cudalize, to_numpy
import numpy as np


class MLPFindModule(nn.Module):
	"""This is the version used in the DNMN paper."""

	def __init__(self):
		super(MLPFindModule, self).__init__()
		self._conv_proj = nn.Conv2d(IMG_DEPTH, ATT_HIDDEN, 1)
		self._wordemb = nn.Parameter(torch.ones(len(FIND_INDEX), ATT_HIDDEN))
		self._conv_mask = nn.Conv2d(ATT_HIDDEN, 1, 1, bias=False)

	def forward(self, features, c):

		if not self.training:
			return self._eval_fwd(features, c)

		proj = self._conv_proj(features)
		B,A,H,W = proj.size()
		M = len(FIND_INDEX)

		# There should be at least N random non-class masks
		c = to_numpy(c)
		nc = set(c)
		N = max(10, len(nc))
		while len(nc) < N:
			nc.add(np.random.randint(M))
		nc = list(nc)
		c = [ nc.index(Ci) for Ci in c ]
		nc = cudalize(torch.tensor(nc))
		c = cudalize(torch.tensor(c))

		proj = proj.unsqueeze(0) #[1,B,A,H,W]
		wemb = self._wordemb[nc].view(N,1,A,1,1)

		attended = F.relu(proj*wemb) #[N,B,A,H,W]
		attended = attended.view(-1,A,H,W)
		mask = self._conv_mask(attended).view(N,B,1,H,W)

		mask = torch.sigmoid(mask)
		total = torch.sum(mask, 0) #[B,1,H,W]
		ones = cudalize(torch.ones([]))
		total = torch.max(total, ones)
		mask = mask[c, torch.arange(B)] #[B,1,H,W]

		mask_train = (mask/total).view(B,-1).mean(1)
		return mask_train, mask

	def _eval_fwd(self, features, c):
		proj = self._conv_proj(features) # [B,A,H,W]
		B, A = proj.size()[:2]
		wemb = self._wordemb[c].view(B,A,1,1)
		attended = F.relu(proj*wemb)
		mask = self._conv_mask(attended)
		return torch.sigmoid(mask)


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
				# This version uses post-sigmoid competition
				mask_all = torch.sigmoid(self._conv(features))
				mask = mask_all[torch.arange(B), c].unsqueeze(1)
				mask_against = (mask_all.sum(1, keepdim=True) - mask) / (B-1)
				mask_train = mask / (1. + mask_against)
				mask_train = mask_train.view(B,-1).mean(1)
				return mask_train, mask
			else:
				# This version uses pre-sigmoid competition (has troubles w/ neg. values)
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


class ClassifyModule(nn.Module):

	def __init__(self):
		super(ClassifyModule, self).__init__()
		self._linear = nn.Linear(IMG_DEPTH, len(ANSWER_INDEX))

	def forward(self, mask, features):
		B,C,H,W = features.size()
		mask_total = mask.view(B,-1).sum(1, keepdim=True) + 1e-10
		mask_normalized = mask / mask_total
		attended = (mask_normalized*features).view(B,C,-1).sum(2)
		return self._linear(attended)


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
