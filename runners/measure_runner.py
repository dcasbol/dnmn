from .base import Runner
from model import Measure
from loaders import MeasureLoader
from misc.util import cudalize

class MeasureRunner(Runner):

	def _get_model(self):
		return Measure(dropout=self._dropout)

	def _get_loader(self, **kwargs):
		return MeasureLoader(**kwargs)

	def _forward(self, batch_data):
		hmap, instance, label = cudalize(*batch_data)
		output = self._model[instance](hmap)
		return dict(
			output = output,
			label  = label
		)
		