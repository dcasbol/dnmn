import torch
from modules import Find, Describe, Measure, QuestionEncoder, GaugeFind
from model import NMN
from runners.base import Runner
from misc.visualization import MapVisualizer
from misc.util import cudalize, cudalize_dict, to_tens, DEVICE
from loaders import EncoderLoader, DescribeLoader, MeasureLoader, NMNLoader
from loaders import GaugeFindLoader


class FindRunner(Runner):

	def __init__(self, visualize=0, **kwargs):
		super(FindRunner, self).__init__(**kwargs)
		self._visualize = visualize
		assert visualize == 0, 'Visualization not implemented yet.'
		if visualize > 0:
			print('WARNING: Find training running with visualization ON')
			self._vis = MapVisualizer(visualize)

	def _get_model(self):
		return GaugeFind(dropout=self._dropout)

	def _loader_class(self):
		return GaugeFindLoader

	def _forward(self, batch_data):
		features, inst_1, inst_2, yesno, label = cudalize(*batch_data[:5])
		pred = self._model(features, inst_1, inst_2, yesno)
		return dict(
			output = pred,
			label  = label
		)

	def _preview(self, mean_loss):
		super(FindRunner, self)._preview(mean_loss)
		if self._visualize > 0:
			keys   = ['hmap', 'label_str', 'input_set', 'input_id']
			values = [ self._result[k] for k in keys ]
			self._vis.update(*values)

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

class MeasureRunner(Runner):

	def _get_model(self):
		return Measure(dropout=self._dropout)

	def _loader_class(self):
		return MeasureLoader

	def _forward(self, batch_data):
		hmap, instance, label = cudalize(*batch_data)
		output = self._model[instance](hmap)
		return dict(
			output = output,
			label  = label
		)

class UncachedRunner(Runner):

	def __init__(self, find_pt='find.pt', **kwargs):
		super(UncachedRunner, self).__init__(**kwargs)
		self._find = Find()
		self._find.load_state_dict(torch.load(find_pt, map_location='cpu'))
		self._find = cudalize(self._find)
		self._find.eval()

	def _loader_class(self):
		return NMNLoader

	def _forward(self, batch_data):
		features  = cudalize(batch_data['features'])
		root_inst = cudalize(batch_data['root_inst'])
		instances = cudalize(batch_data['find_inst'])

		with torch.no_grad():
			hmaps = generate_hmaps(self._find, instances, features)

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

class EncoderRunner(Runner):

	def _get_model(self):
		return QuestionEncoder(dropout=self._dropout)

	def _loader_class(self):
		return EncoderLoader

	def _forward(self, batch_data):
		question, length, label = cudalize(*batch_data)
		output = self._model(question, length)
		return dict(
			output = output,
			label  = label
		)

class NMNRunner(Runner):

	def __init__(self, find_pt=None, **kwargs):
		super(NMNRunner, self).__init__(**kwargs)
		self._keys = ['features', 'question', 'length', 'yesno', 'root_inst', 'find_inst']
		if find_pt is not None:
			self._model.load_module(Find.NAME, find_pt)
			find_params = [ hash(p) for p in self._model._find.parameters() ]
			parameters = [ p for p in self._model.parameters() if hash(p) not in find_params ]
			lr, weight_decay = [ self._opt.defaults[k] for k in ['lr', 'weight_decay'] ]
			self._opt = torch.optim.Adam(parameters,
				lr=lr, weight_decay=weight_decay)

	def _get_model(self):
		return NMN(dropout=self._dropout)

	def _loader_class(self):
		return NMNLoader

	def _get_nmn_data(self, batch_data):
		return [ batch_data[k] for k in self._keys ]

	def _forward(self, batch_data):
		batch_data = cudalize_dict(batch_data, exclude='find_inst')
		batch_data['find_inst'] = cudalize(*batch_data['find_inst'])
		nmn_data = self._get_nmn_data(batch_data)
		pred = self._model(*nmn_data)
		return dict(
			output = pred,
			label  = batch_data['label']
		)
