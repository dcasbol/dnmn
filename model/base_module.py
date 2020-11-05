import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.indices import UNK_ID
from misc.util import DEVICE


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
				continue
			loss = self._loss_fn(pred[mask], y[mask])
			loss_list.append(loss)
		if loss_list == []:
			return torch.zeros([], device=DEVICE, requires_grad=True)
		return sum(loss_list)

	def save(self, filename):
		torch.save(self.state_dict(), filename)
		print('{} saved at {!r}'.format(self.NAME, filename))

	def load(self, filename):
		self.load_state_dict(torch.load(filename, map_location='cpu'))
		print('{} loaded from {!r}'.format(self.NAME, filename))