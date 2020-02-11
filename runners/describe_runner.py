from .base import Runner
from modules import Describe
from loaders import DescribeLoader
from misc.util import cudalize

class DescribeRunner(Runner):

	def _get_model(self):
		return Describe(dropout=self._dropout)

	def _loader_class(self):
		return DescribeLoader

	def _forward(self, batch_data):
		attended, instance, label = cudalize(*batch_data)
		output = self._model[instance](attended)
		return dict(
			output = output,
			label  = label
		)
		