import torch.nn as nn
import torch.nn.functional as F
from .instance_module import InstanceModule
from misc.indices import FIND_INDEX
from misc.constants import *


class Find(InstanceModule):
	"""find[c] convolves every position in the input image with a weight vector
	(distinct for each c) to produce a heatmap or unnormalized attention. [...]
	the output of the module find[dog] is a matrix whose entries should be large
	in regions of the image containing dogs, and small everywhere else."""

	NAME = 'find'

	def __init__(self, modular=False, bias=False, **kwargs):
		super(Find, self).__init__(dropout=0, **kwargs)
		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=bias)
		if modular:
			self._act_fn = nn.Sigmoid()
		else:
			self._act_fn = nn.ReLU()
			self._conv.weight.data.fill_(0.01)

	def forward(self, features):
		c = self._get_instance()
		B,D,H,W = features.size()
		kernel = self._conv.weight[c]
		group_feats = features.contiguous().view(1,B*D,H,W)
		hmap = F.conv2d(group_feats, kernel, groups=B)
		hmap = self._act_fn(hmap).view(B,1,H,W)
		return hmap

	def loss(self, pred, target):
		raise NotImplementedError
