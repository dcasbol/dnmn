from .base import Runner
from model import Describe
from loaders import DescribeLoader
from misc.util import cudalize

class DescribeRunner(Runner):

	def _get_model(self):
		return Describe(dropout=self._dropout)

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