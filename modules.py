import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX
from misc.constants import *
from misc.util import cudalize
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
		c = c.detach().cpu().numpy()
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

	def __init__(self):
		super(FindModule, self).__init__()
		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)

	def forward(self, features, c):

		if self.training:
			h = self._conv(features)
			mean = h.mean(1, keepdim=True)
			B = h.size(0)
			h = h[torch.arange(B), c].unsqueeze(1)
			mask_train = torch.sigmoid(h-mean).view(B,-1).mean()
			mask = torch.sigmoid(h)
			return mask_train, mask

			x = torch.sigmoid(self._conv(features))
			total = x.sum(1, keepdim=True)
			B = x.size(0)
			x = x[torch.arange(B), c].unsqueeze(1)
			mask_train = x / (1. + total - x)
			mask_train = x.view(B,-1).mean(1)
			return mask_train, x
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


class AndModule(nn.Module):

	def forward(self, mask_a, mask_b):
		#return torch.min(mask_a, mask_b)
		return mask_a*mask_b