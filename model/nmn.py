import torch
import torch.nn.functional as F
from misc.util import cudalize, DEVICE, generate_hmaps
from .base_module import BaseModule
from model import Find, Describe, Measure, QuestionEncoder
from misc.indices import ANSWER_INDEX, UNK_ID


class NMN(BaseModule):

	NAME = 'nmn'

	def __init__(self, dropout=0, modular=False):
		super(NMN, self).__init__()
		self._find     = Find()
		self._describe = Describe(dropout=dropout)
		self._measure  = Measure(dropout=dropout)
		self._encoder  = QuestionEncoder(dropout=dropout)
		self._modnames = [ m.NAME for m in [ Find, Describe, Measure, QuestionEncoder ] ]
		self._modular  = modular

	def loss(self, x, labels):
		loss_list  = list()
		batch_size = x.size(0)
		B_idx = torch.arange(batch_size)
		for y in labels.t():
			mask = y != UNK_ID
			if not mask.any():
				break
			p  = x[B_idx, y][mask]
			ce = -(p+1e-10).log().sum()
			loss_list.append(ce)
		if loss_list == []:
			return torch.zeros([], device=DEVICE, requires_grad=True)
		return sum(loss_list) / batch_size

	def forward(self, features, question, length, yesno, root_inst, find_inst):

		# Drop the same channels for all Find instances
		features = self._dropout2d(features)

		hmaps = generate_hmaps(self._find, find_inst, features)

		root_pred = torch.empty(yesno.size(0), len(ANSWER_INDEX), device=DEVICE)

		yesno_hmaps = hmaps[yesno]
		if yesno_hmaps.size(0) > 0:
			yesno_inst = root_inst[yesno]
			yesno_ans = self._measure[yesno_inst](yesno_hmaps)
			root_pred[yesno] = yesno_ans

		other = ~yesno
		other_hmaps = hmaps[other]
		if other_hmaps.size(0) > 0:
			other_inst = root_inst[other]
			other_fts  = features[other]
			other_ans  = self._describe[other_inst](other_hmaps, other_fts)
			root_pred[other] = other_ans

		root_pred = root_pred
		enc_pred  = self._encoder(question, length)

		if self._modular:
			return (root_pred+enc_pred).softmax(1)

		root_pred = root_pred.softmax(1)
		enc_pred  = enc_pred.softmax(1)
		return (root_pred*enc_pred + 1e-30).sqrt()

	def load_module(self, module_name, filename):
		assert module_name in self._modnames
		getattr(self, '_'+module_name).load(filename)

	def save_module(self, module_name, filename):
		assert module_name in self._modnames
		getattr(self, '_'+module_name).save(filename)
