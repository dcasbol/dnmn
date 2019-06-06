import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import MODULE_INDEX
from misc.constants import *

class FindModule(nn.Module):

	def __init__(self):
		super(FindModule, self).__init__()
		self._conv = nn.Conv2d(IMG_DEPTH, len(MODULE_INDEX), 1, bias=True)

	def forward(self, features, c):
		""" Conflict:
		NMN paper says <<convolves every position in the input
		image with a weight vector (distinct for each c)
		to produce a heatmap of unnormalized attention>>

		THAT IS NOT WHAT I SEE IN CODE. They do several steps
		and then use local normalization (sigmoid). """

		# 1. 1x1 conv. in_depth --> hiddens
		# 2. select word embedding
		# 3. eltwise sum to projection + ReLU
		# 4. 1x1 conv. hiddens --> 1
		# 5. local (opt. global) normalization (sgm/softmax)
		# 6. Power layer: y = (shift + scale * x)^power

		if self.training:
			x = torch.sigmoid(self._conv(features))
			tot = torch.sum(x, 1, keepdim=True)
			bs, _, height, width = features.size()
			x = x[torch.arange(bs), c].view(bs, 1, height, width)
			x = x / (tot + 1e-10)
		else:
			bs = c.size(0)
			_, in_ch, height, width = self._conv.weight.size()
			k = self._conv.weight[c].view(bs, in_ch, height, width)
			#bias = self._conv.bias[c]
			x = torch.sigmoid(F.conv2d(features, k))
		return x

class MLPFindModule(nn.Module):

	def __init__(self):
		super(MLPFindModule, self).__init__()
		self._conv_proj = nn.Conv2d(IMG_DEPTH, ATT_HIDDEN, 1)
		self._wordvec = nn.Parameter(torch.rand(len(MODULE_INDEX), ATT_HIDDEN))
		self._conv_mask = nn.Conv2d(ATT_HIDDEN, 1, 1)