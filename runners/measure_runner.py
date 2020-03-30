from .base import Runner
from model import Measure
from loaders import MeasureLoader
from misc.util import cudalize
from misc.constants import *

class MeasureRunner(Runner):

	def __init__(self, hidden_size=HIDDEN_SIZE, hidden_dropout=0, **kwargs):
		self._hidden_size    = hidden_size
		self._hidden_dropout = hidden_dropout
		super(MeasureRunner, self).__init__(**kwargs)

	def _get_model(self):
		return Measure(
			dropout        = self._dropout,
			hidden_size    = self._hidden_size,
			hidden_dropout = self._hidden_dropout
		)

	def _get_loader(self, **kwargs):
		return MeasureLoader(prior=self._modular, **kwargs)

	def _forward(self, batch_data):
		hmap, instance, label = cudalize(*batch_data[:3])
		prior = cudalize(batch_data[-1]) if self._modular else None
		output = self._model[instance](hmap, prior=prior)
		return dict(
			output = output,
			label  = label
		)