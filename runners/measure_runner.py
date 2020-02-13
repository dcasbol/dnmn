from .base import Runner
from model import Measure
from loaders import MeasureLoader
from misc.util import cudalize

class MeasureRunner(Runner):

	def _get_model(self):
		return Measure(dropout=self._dropout)

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