import torch
from .base import Runner
from model import Find, Describe, Measure
from loaders import NMNLoader
from misc.util import cudalize, generate_hmaps

class UncachedRunner(Runner):

	def __init__(self, find_pt='find.pt', modular=False, **kwargs):
		super(UncachedRunner, self).__init__(**kwargs)
		self._find = Find(modular=modular)
		self._find.load_state_dict(torch.load(find_pt, map_location='cpu'))
		self._find = cudalize(self._find)
		self._find.eval()
		self._modular = modular

	def _loader_class(self):
		return NMNLoader

	def _forward(self, batch_data):
		features  = cudalize(batch_data['features'])
		root_inst = cudalize(batch_data['root_inst'])
		inst = cudalize(*batch_data['find_inst'])
		inst = (inst,) if isinstance(inst, torch.Tensor) else inst

		with torch.no_grad():
			hmaps = generate_hmaps(self._find, inst, features, self._modular)

		return dict(
			output = self._model[root_inst](hmaps, features),
			label  = cudalize(batch_data['label'])
		)

class DescribeRunnerUncached(UncachedRunner):

	def _get_model(self):
		return Describe(dropout=self._dropout)

class MeasureRunnerUncached(UncachedRunner):

	def _get_model(self):
		return Measure(dropout=self._dropout)
