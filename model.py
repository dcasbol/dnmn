import torch
import torch.nn.functional as F
from misc.util import cudalize, DEVICE, USE_CUDA, generate_hmaps
from modules import Find, Describe, Measure, QuestionEncoder, BaseModule
from misc.indices import ANSWER_INDEX


class NMN(BaseModule):

	NAME = 'nmn'

	def __init__(self, dropout=0):
		super(NMN, self).__init__()
		self._find     = Find()
		self._describe = Describe(dropout=dropout)
		self._measure  = Measure(dropout=dropout)
		self._encoder  = QuestionEncoder(dropout=dropout)
		self._modnames = [ m.NAME for m in [ Find, Describe, Measure, QuestionEncoder ] ]

	def loss(self, x, labels):
		loss_list = list()
		B_idx = torch.arange(x.size(0))
		for y in labels:
			p  = x[B_idx, y]
			ce = -(p+1e-10).log().mean()
			loss_list.append(ce)
		return sum(loss_list)

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

		root_pred = root_pred.softmax(1)
		enc_pred  = self._encoder(question, length).softmax(1)

		return (root_pred*enc_pred + 1e-30).sqrt()

	def load_module(self, module_name, filename):
		assert module_name in self._modnames
		getattr(self, '_'+module_name).load(filename)

	def save_module(self, module_name, filename):
		assert module_name in self._modnames
		getattr(self, '_'+module_name).save(filename)
