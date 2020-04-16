from .base import Runner
from model import Describe
from loaders import DescribeLoader
from misc.util import cudalize
from misc.constants import *

class DescribeRunner(Runner):

	def __init__(self, hidden_size=None, hidden_dropout=0, **kwargs):
		self._hidden_size    = hidden_size
		self._hidden_dropout = hidden_dropout
		super(DescribeRunner, self).__init__(**kwargs)

	def _get_model(self):
		return Describe(
			dropout        = self._dropout,
			hidden_size    = self._hidden_size,
			hidden_dropout = self._hidden_dropout
		)

	def _get_loader(self, **kwargs):
		return DescribeLoader(prior=self._modular, **kwargs)

	def _forward(self, batch_data):
		attended, instance, label = cudalize(*batch_data[:3])
		prior = cudalize(batch_data[-1]) if self._modular else None
		output = self._model[instance](attended, prior=prior)
		return dict(
			output = output,
			label  = label
		)
