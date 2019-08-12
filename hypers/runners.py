from modules import Find, Describe, Measure, QuestionEncoder
from vqa import VQAFindDataset, VQADescribeDataset, VQAMeasureDataset, VQAEncoderDataset
from hypers.base import Runner
from misc.visualization import MapVisualizer
from misc.util import cudalize

class FindRunner(Runner):

	def __init__(self, competition='softmax', visualize=0, dropout=False, **kwargs):
		self._model   = Find(competition=competition, dropout=dropout)
		self._dataset = VQAFindDataset(metadata=True)
		if 'validate' in kwargs:
			assert not kwargs['validate'], "Can't validate Find just yet"
		super(FindRunner, self).__init__(**kwargs)
		self._visualize = visualize
		if visualize > 0:
			self._vis = MapVisualizer(visualize)

	def _forward(self, batch_data):
		features, instance, label_str, input_set, input_id = batch_data
		features, instance = cudalize(features, instance)
		output = self._model[instance](features)
		self._result = dict(
			output    = output,
			hmap      = output,
			label_str = label_str,
			input_set = input_set,
			input_id  = input_id
		)
		return self._result

	def _log_routine(self, epoch, mean_loss):
		super(FindRunner, self)._log_routine(epoch, mean_loss)
		if self._visualize > 0:
			keys   = ['hmap', 'label_str', 'input_set', 'input_id']
			values = [ self._result[k] for k in keys ]
			self._vis.update(*values)

class DescribeRunner(Runner):

	def __init__(self, dropout=False, **kwargs):
		self._model = Describe(dropout=dropout)
		self._dataset = VQADescribeDataset()
		super(DescribeRunner, self).__init__(**kwargs)

	def _forward(self, batch_data):
		mask, features, label, distr = cudalize(*batch_data[:2]+batch_data[3:])
		instance = batch_data[2]
		output = self._model[instance](mask, features)
		return dict(
			output = output,
			label  = label,
			distr  = distr
		)

class MeasureRunner(Runner):

	def __init__(self, dropout=False, **kwargs):
		self._model = Measure(dropout=dropout)
		self._dataset = VQAMeasureDataset()
		super(MeasureRunner, self).__init__(**kwargs)

	def _forward(self, batch_data):
		mask = cudalize(batch_data[0])
		label, distr = cudalize(*batch_data[2:])
		instance = batch_data[1]
		output = self._model[instance](mask)
		return dict(
			output = output,
			label  = label,
			distr  = distr
		)

class EncoderRunner(Runner):

	def __init__(self, dropout=False, embed_size=None, **kwargs):
		self._model = QuestionEncoder(dropout=dropout, embed_size=embed_size)
		self._dataset = VQAEncoderDataset()
		super(EncoderRunner, self).__init__(**kwargs)

	def _forward(self, batch_data):
		question, length, label, distr = cudalize(*batch_data)
		output = self._model(question, length)
		return dict(
			output = output,
			label  = label,
			distr  = distr
		)