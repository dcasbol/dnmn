import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX, QUESTION_INDEX, DESC_INDEX, UNK_ID
from misc.constants import *
from misc.util import DEVICE, attend_features, generate_hmaps
import misc.util as util


class BaseModule(nn.Module):

	def __init__(self, dropout=0):
		super(BaseModule, self).__init__()
		self._dropout = {
			False : lambda x: x,
			True  : lambda x: F.dropout(x, p=dropout, training=self.training)
		}[dropout>0]
		self._dropout2d = {
			False : lambda x: x,
			True  : lambda x: F.dropout2d(x, p=dropout, training=self.training)
		}[dropout>0]
		self._loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

	def loss(self, pred, labels):
		loss_list  = list()
		batch_size = pred.size(0)
		for y in labels.t():
			mask = y != UNK_ID
			if not mask.any():
				break
			loss = self._loss_fn(pred[mask], y[mask])
			loss_list.append(loss)
		if loss_list == []:
			return torch.zeros([], device=DEVICE, requires_grad=True)
		return sum(loss_list) / batch_size

	def save(self, filename):
		torch.save(self.state_dict(), filename)
		print('{} saved at {!r}'.format(self.NAME, filename))

	def load(self, filename):
		self.load_state_dict(torch.load(filename, map_location='cpu'))
		print('{} loaded from {!r}'.format(self.NAME, filename))


class InstanceModule(BaseModule):
	"""Module with [] overloaded to follow nomenclature as in paper:
	Find[inst](features)"""

	def __init__(self, **kwargs):
		super(InstanceModule, self).__init__(**kwargs)
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
	"""find[c] convolves every position in the input image with a weight vector
	(distinct for each c) to produce a heatmap or unnormalized attention. [...]
	the output of the module find[dog] is a matrix whose entries should be large
	in regions of the image containing dogs, and small everywhere else."""

	NAME = 'find'

	def __init__(self, modular=False, **kwargs):
		super(Find, self).__init__(dropout=0, **kwargs)
		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)
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


class Describe(InstanceModule):
	"""First computes an average over image features weighted by the attention,
	then passes this averaged feature vector through a single fully-connected layer."""

	NAME = 'describe'

	def __init__(self, **kwargs):
		super(Describe, self).__init__(**kwargs)
		self._descr = list()
		for i in range(len(DESC_INDEX)):
			layer = nn.Linear(IMG_DEPTH, len(ANSWER_INDEX))
			setattr(self, '_descr_%d' % i, layer)
			self._descr.append(layer)

	def forward(self, hmap_or_attended, features=None):
		if features is None:
			attended = hmap_or_attended
		else:
			attended = attend_features(features, hmap_or_attended)

		attended = self._dropout(attended)
		attended = attended.unsqueeze(1).unbind(0)
		instance = self._get_instance()
		preds = list()
		for att, inst in zip(attended, instance):
			preds.append(self._descr[inst](att))

		return torch.cat(preds)


class Measure(InstanceModule):
	""" measure[c] takes an attention alone and maps it to a distribution over labels.
	hmap -> FC -> ReLU -> FC -> Softmax -> ans """

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


class QuestionEncoder(BaseModule):
	"""Single-layer LSTM with 1024 units. The question modeling component predicts
	a distribution over the set of answers."""

	NAME = 'encoder'

	def __init__(self, dropout=0):
		super(QuestionEncoder, self).__init__(dropout=dropout)
		self._wemb = nn.Embedding(len(QUESTION_INDEX), EMBEDDING_SIZE)
		self._lstm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_UNITS)
		self._final = nn.Linear(HIDDEN_UNITS, len(ANSWER_INDEX))

	def forward(self, question, length):
		B = length.size(0)
		embed = self._wemb(question)
		hidden = self._lstm(embed)[0][length-1, torch.arange(B)]
		return self._final(self._dropout(hidden))


class GaugeFind(BaseModule):
	"""This module integrates a Gauge sub-module, which acts as an utility function
	for training the Find module"""

	NAME = 'gauge-find'

	def __init__(self, dropout=0, modular=False, **kwargs):
		super(GaugeFind, self).__init__(dropout=dropout)
		self._classifier = nn.Sequential(
			nn.Linear(IMG_DEPTH + MASK_WIDTH**2, 64, bias=False),
			nn.Linear(64, len(ANSWER_INDEX))
		)
		self._find = Find(**kwargs)
		self._forced_dropout = lambda x: F.dropout(x, p=0.3, training=True)
		self._modular = modular

	def forward(self, features, inst_1, inst_2, yesno):

		features = self._dropout2d(features)

		instances = [inst_1, inst_2] if (inst_2>0).any() else [inst_1]
		hmap = generate_hmaps(self._find, instances, features, self._modular)

		B = hmap.size(0)
		yesno = yesno.view(B,1).float()
		attended  = attend_features(features, hmap)*(1.-yesno)
		hmap_flat = hmap.view(B,-1)*yesno
		x = torch.cat([attended, hmap_flat], 1)
		if self.training:
			pred = self._classifier(self._forced_dropout(x))
			return pred
		else:
			x = x.view(B,1,-1).expand(-1,20,-1)
			preds = self._classifier(self._forced_dropout(x)).softmax(dim=2)
			mean = preds.mean(1)
			idx  = mean.argmax(1)
			var  = preds[torch.arange(B),:,idx].var(1)
			return mean, var

	def save(self, filename):
		self._find.save(filename)

	def load(self, filename):
		self._find.load(filename)
