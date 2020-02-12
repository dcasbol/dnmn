import torch
from .base import Runner
from model import NMN
from loaders import NMNLoader
from misc.util import cudalize, cudalize_dict

class NMNRunner(Runner):

	def __init__(self, find_pt=None, modular=False, **kwargs):
		self._modular = modular
		super(NMNRunner, self).__init__(**kwargs)
		self._keys = ['features', 'question', 'length', 'yesno', 'root_inst', 'find_inst']
		if find_pt is not None:
			self._model.load_module(Find.NAME, find_pt)
			find_params = [ hash(p) for p in self._model._find.parameters() ]
			parameters = [ p for p in self._model.parameters() if hash(p) not in find_params ]
			lr, weight_decay = [ self._opt.defaults[k] for k in ['lr', 'weight_decay'] ]
			self._opt = torch.optim.Adam(parameters,
				lr=lr, weight_decay=weight_decay)

	def _get_model(self):
		return NMN(dropout=self._dropout, modular=self._modular)

	def _get_loader(self, **kwargs):
		return NMNLoader(**kwargs)

	def _get_nmn_data(self, batch_data):
		return [ batch_data[k] for k in self._keys ]

	def _forward(self, batch_data):
		batch_data = cudalize_dict(batch_data, exclude='find_inst')
		inst = cudalize(*batch_data['find_inst'])
		batch_data['find_inst'] = (inst,) if isinstance(inst, torch.Tensor) else inst
		nmn_data = self._get_nmn_data(batch_data)
		pred = self._model(*nmn_data)
		return dict(
			output = pred,
			label  = batch_data['label']
		)
