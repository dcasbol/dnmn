from .base import Runner
from modules import GaugeFind
from loaders import GaugeFindLoader
from misc.util import cudalize
from misc.visualization import MapVisualizer

class FindRunner(Runner):

	def __init__(self, visualize=0, modular=False, **kwargs):
		self._modular   = modular
		super(FindRunner, self).__init__(**kwargs)
		self._visualize = visualize
		assert visualize == 0, 'Visualization not implemented yet.'
		if visualize > 0:
			print('WARNING: Find training running with visualization ON')
			self._vis = MapVisualizer(visualize)

	def _get_model(self):
		return GaugeFind(dropout=self._dropout, modular=self._modular)

	def _loader_class(self):
		return GaugeFindLoader

	def _forward(self, batch_data):
		features, inst_1, inst_2, yesno, label = cudalize(*batch_data[:5])
		pred = self._model(features, inst_1, inst_2, yesno)
		result = dict(
			output = pred,
			label = label
		)
		if not self._model.training:
			result['output'], result['var'] = pred
		return result

	def _preview(self, mean_loss):
		super(FindRunner, self)._preview(mean_loss)
		if self._visualize > 0:
			keys   = ['hmap', 'label_str', 'input_set', 'input_id']
			values = [ self._result[k] for k in keys ]
			self._vis.update(*values)

	@property
	def last_var(self):
		return self._logger.last('var')

	@property
	def last_agreement(self):
		return self._logger.last('agreement')
		