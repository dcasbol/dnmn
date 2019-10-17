import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import FIND_INDEX, ANSWER_INDEX, QUESTION_INDEX, DESC_INDEX, NULL_ID
from misc.constants import *
from misc.util import to_numpy, attend_features, USE_CUDA, generate_hmaps
from time import time


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

	def loss(self, pred, target):
		return self._loss_fn(pred, target)

	def save(self, filename):
		torch.save(self.state_dict(), filename)
		print('{} saved at {!r}'.format(self.NAME, filename))

	def load(self, filename):
		self.load_state_dict(torch.load(filename, map_location='cpu'))
		if USE_CUDA:
			self.cuda()
		print('{} loaded from {!r}'.format(self.NAME, filename))


class InstanceModule(BaseModule):

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
	"""This module corresponds to the original 'attend' in the NMN paper."""

	NAME = 'find'

	def __init__(self, activation='relu', **kwargs):
		super(Find, self).__init__(dropout=0, **kwargs)
		self._conv = nn.Conv2d(IMG_DEPTH, len(FIND_INDEX), 1, bias=False)
		self._conv.weight.data.fill_(0.01)
		self._act_fn = dict(
			elu   = lambda x: F.elu(x) + 1.0,
			srelu = lambda x: F.softsign(x)*0.5 + 0.5 + x.relu(),
			relu  = lambda x: x.relu()
		)[activation]

	def forward(self, features):
		c = self._get_instance()
		B,D,H,W = features.size()
		kernel = self._conv.weight[c]
		group_feats = features.contiguous().view(1,B*D,H,W)
		hmap = F.conv2d(group_feats, kernel, groups=B)
		hmap = self._act_fn(hmap)
		return hmap.view(B,1,H,W)

	def loss(self, pred, target):
		raise NotImplementedError


class Describe(InstanceModule):
	""" From 1st NMN article: It first computes an average over image features
	weighted by the attention, then passes this averaged feature vector through
	a single fully-connected layer. """
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


class DescribeBeta(InstanceModule):

	NAME = 'describe'

	def __init__(self, **kwargs):
		super(DescribeBeta, self).__init__(**kwargs)
		self._full_layer = nn.Linear(len(DESC_INDEX)*IMG_DEPTH, len(ANSWER_INDEX))
		self._weights = self._full_layer.weight.view(len(DESC_INDEX), IMG_DEPTH, len(ANSWER_INDEX))
		self._bias    = self._full_layer.bias.view(1, len(ANSWER_INDEX))

	def forward(self, hmap_or_attended, features=None):
		if features is None:
			attended = hmap_or_attended
		else:
			attended = attend_features(features, hmap_or_attended)

		attended = self._dropout(attended)
		instance = self._get_instance()
		weights  = self._weights[instance]
		return attended*weights + self._bias


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


class QuestionEncoder(BaseModule):

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


class InstancePredictor(nn.Module):

	def __init__(self):
		super(InstancePredictor, self).__init__()
		self._classifier = nn.Linear(IMG_DEPTH, len(FIND_INDEX))
		self._loss_fn = nn.CrossEntropyLoss(reduction='sum')

	def forward(self, attended):
		return self._classifier(attended)

	def loss(self, pred, instance):
		return self._loss_fn(pred, instance)


class GaugeFind(BaseModule):

	NAME = 'gauge-find'

	def __init__(self, dropout=0, **kwargs):
		super(GaugeFind, self).__init__(dropout=dropout)
		self._classifier = nn.Sequential(
			nn.Linear(IMG_DEPTH + MASK_WIDTH**2, 64, bias=False),
			nn.Linear(64, len(ANSWER_INDEX))
		)
		self._find = Find(**kwargs)

	def forward(self, features, inst_1, inst_2, yesno):

		features = self._dropout2d(features)

		hmap = generate_hmaps(self._find, [inst_1, inst_2], features)

		B = hmap.size(0)
		yesno = yesno.view(B,1).float()
		attended  = attend_features(features, hmap)*(1.-yesno)
		hmap_flat = hmap.view(B,-1)*yesno
		x = torch.cat([attended, hmap_flat], 1)
		pred = self._classifier(self._dropout(x))
		return pred

	def save(self, filename):
		self._find.save(filename)

	def load(self, filename):
		self._find.load(filename)
