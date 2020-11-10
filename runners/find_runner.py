from .base import Runner
from model import GaugeFind
from loaders import GaugeFindLoader
from misc.util import cudalize
from misc.visualization import MapVisualizer

class FindRunner(Runner):

	def __init__(self, visualize=0, softmax_attn=False, bias=False, hq_gauge=False, **kwargs):
		self._softmax_attn = softmax_attn
		self._bias         = bias
		self._hq_gauge     = hq_gauge
		super(FindRunner, self).__init__(**kwargs)
		self._visualize = visualize
		assert visualize == 0, 'Visualization not implemented yet'
		if visualize > 0:
			print('WARNING: Find training running with visualization ON')
			self._vis = MapVisualizer(visualize)

	def _get_model(self):
		return GaugeFind(
			dropout      = self._dropout,
			modular      = self._modular,
			softmax_attn = self._softmax_attn,
			bias         = self._bias,
			hq           = self._hq_gauge
		)

	def _get_loader(self, **kwargs):
		return GaugeFindLoader(prior=self._modular, inst=self._hq_gauge, **kwargs)

	def _forward(self, batch_data):
		features, inst_1, inst_2, yesno, label = cudalize(*batch_data[:5])
		qinst = cudalize(batch_data[-2]) if self._hq_gauge else None
		prior = cudalize(batch_data[-1]) if self._modular else None
		pred = self._model(features, inst_1, inst_2, yesno, qinst, prior)
		result = dict(
			output = pred,
			label = label
		)
		if not self._model.training:
			result['output'], result['var'] = pred
		if self._visualize > 0:
			self._result = result
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
