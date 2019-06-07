import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import MODULE_INDEX
from misc.constants import *
from misc.util import cudalize
import numpy as np


class FindModule(nn.Module):

	def __init__(self):
		super(FindModule, self).__init__()
		self._conv_proj = nn.Conv2d(IMG_DEPTH, ATT_HIDDEN, 1)
		self._conv = nn.Conv2d(ATT_HIDDEN, len(MODULE_INDEX), 1)

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
			x = F.relu(self._conv_proj(features))
			x = torch.sigmoid(self._conv(x))
			tot = torch.sum(x, 1, keepdim=True)
			B = x.size(0)
			x = x[torch.arange(B), c].unsqueeze(1)
			x = x / (tot + 1e-10)
		else:
			_, in_ch, height, width = self._conv.weight.size()
			k = self._conv.weight[c].view(-1, in_ch, height, width)
			x = torch.sigmoid(F.conv2d(features, k))
		return x

class MLPFindModule(nn.Module):

	def __init__(self):
		super(MLPFindModule, self).__init__()
		self._conv_proj = nn.Conv2d(IMG_DEPTH, ATT_HIDDEN, 1)
		self._wordemb = nn.Parameter(torch.rand(len(MODULE_INDEX), ATT_HIDDEN))
		self._conv_mask = nn.Conv2d(ATT_HIDDEN, 1, 1)
		self._print = True

	def forward(self, features, c):

		proj = self._conv_proj(features)
		B,A,H,W = proj.size()
		M = len(MODULE_INDEX)

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

		attended = F.relu(proj+wemb) #[M,B,A,H,W]
		attended = attended.view(-1,A,H,W)
		mask = torch.sigmoid(self._conv_mask(attended)).view(N,B,1,H,W)

		total = torch.sum(mask, 0) #[B,1,H,W]
		mask = mask[c, torch.arange(B)] #[B,1,H,W]
		return mask / (total + 1e-10)