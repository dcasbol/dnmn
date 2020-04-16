import os
import pickle
from .base_module import BaseModule
from model import Find, Describe, Measure, QuestionEncoder
from misc.util import cudalize, DEVICE, generate_hmaps
from misc.indices import ANSWER_INDEX


class ResultObject:
	pass

class NMNPretrained(BaseModule):
	"""Collect all results from modular hyperparameter optimization and
	put it all together for the final evaluation on the held-out set.
	This is intended to work for modular=True."""

	NAME = 'nmn'

	def __init__(self, hpo_dir):
		super(NMNPretrained, self).__init__()
		softmax_attn = False
		ignored_params = ['learning_rate', 'batch_size', 'weight_decay', 'dropout']
		for name in ['encoder','find','describe','measure']:
			pt_file  = os.path.join(hpo_dir, '{0}/{0}-hpo-best.pt'.format(name))
			res_file = os.path.join(hpo_dir, '{0}/{0}-res.dat'.format(name))

			with open(res_file, 'rb') as fd:
				res = pickle.load(fd)
			kwargs = res.x_iters[res.best_eval]
			kwargs = { k:v for k,v in kwargs.items() if k not in ignored_params }
			if name == 'find':
				kwargs['modular'] = True
			if name == 'find' and 'softmax_attn' in kwargs:
				softmax_attn = kwargs['softmax_attn']
				del kwargs['softmax_attn']
			if name == 'describe':
				kwargs['softmax_attn'] = softmax_attn

			module = dict(
				encoder  = QuestionEncoder,
				find     = Find,
				describe = Describe,
				measure  = Measure
			)[name](**kwargs)
			module.load(pt_file)
			setattr(self, '_'+name, module)

	def forward(self, features, question, length, yesno, root_inst, find_inst):
		""" This will only work in forward mode """

		hmaps = generate_hmaps(self._find, find_inst, features, modular=True)

		pred = self._encoder(question, length)

		yesno_hmaps = hmaps[yesno]
		if yesno_hmaps.size(0) > 0:
			yesno_inst = root_inst[yesno]
			yesno_pred = self._measure[yesno_inst](yesno_hmaps)
			pred[yesno] += yesno_pred

		other = ~yesno
		other_hmaps = hmaps[other]
		if other_hmaps.size(0) > 0:
			other_inst = root_inst[other]
			other_fts  = features[other]
			other_pred = self._describe[other_inst](other_hmaps, other_fts)
			pred[other] += other_pred

		return pred