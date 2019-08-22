from modules import Find, Describe, Measure, QuestionEncoder
from model import NMN
from runners.base import Runner
from misc.visualization import MapVisualizer
from misc.util import cudalize, cudalize_dict, to_tens
from loaders import EncoderLoader, FindLoader, DescribeLoader, MeasureLoader, NMNLoader


class FindRunner(Runner):

	def __init__(self, competition='softmax', visualize=0, dropout=False, **kwargs):
		self._model   = Find(competition=competition, dropout=dropout)
		if 'validate' in kwargs:
			assert not kwargs['validate'], "Can't validate Find just yet"
		super(FindRunner, self).__init__(**kwargs)
		self._visualize = visualize
		if visualize > 0:
			self._vis = MapVisualizer(visualize)

	def _loader_class(self):
		return FindLoader

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

	def _log_routine(self, mean_loss):
		super(FindRunner, self)._log_routine(epoch, mean_loss)
		if self._visualize > 0:
			keys   = ['hmap', 'label_str', 'input_set', 'input_id']
			values = [ self._result[k] for k in keys ]
			self._vis.update(*values)

class DescribeRunner(Runner):

	def __init__(self, dropout=False, **kwargs):
		self._model = Describe(dropout=dropout)
		super(DescribeRunner, self).__init__(**kwargs)

	def _loader_class(self):
		return DescribeLoader

	def _forward(self, batch_data):
		mask, features, label, distr = cudalize(*batch_data[:2]+batch_data[3:])
		instance = batch_data[2]
		output = self._model[instance](mask, features)
		return dict(
			output = output,
			label  = label,
			distr  = distr
		)

class DescribeRunnerUncached(Runner):

	def __init__(self, dropout=False, find_pt='find.pt', **kwargs):
		self._model = Describe(dropout=dropout)
		super(DescribeRunnerUncached, self).__init__(**kwargs)
		self._find = Find(competition=None)
		self._find.load_state_dict(torch.load(find_pt, map_location='cpu'))
		self._find = cudalize(self._find)

	def _loader_class(self):
		return NMNLoader

	def _forward(self, batch_data):
		features  = cudalize(batch_data['features'])
		root_inst = cudalize(batch_data['root_inst'])
		find_inst = [ to_tens(inst, 'long') for inst in batch_data['find_inst'] ]

		maps = list()
		for f, inst in zip(features_list, find_inst):
			f = f.expand(len(inst), -1, -1, -1)
			m = self._find[inst](f).prod(0, keepdim=True)
			maps.append(m)
		maps = torch.cat(maps)

		return dict(
			output = self._model[root_inst](maps, features),
			label  = batch_data['label'],
			distr  = batch_data['distr']
		)

class MeasureRunner(Runner):

	def __init__(self, dropout=False, **kwargs):
		self._model = Measure(dropout=dropout)
		super(MeasureRunner, self).__init__(**kwargs)

	def _loader_class(self):
		return MeasureLoader

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
		super(EncoderRunner, self).__init__(**kwargs)

	def _loader_class(self):
		return EncoderLoader

	def _forward(self, batch_data):
		question, length, label, distr = cudalize(*batch_data)
		output = self._model(question, length)
		return dict(
			output = output,
			label  = label,
			distr  = distr
		)

class NMNRunner(Runner):

	def __init__(self, dropout=False, **kwargs):
		self._model = NMN(dropout=dropout)
		super(NMNRunner, self).__init__(**kwargs)

	def _loader_class(self):
		return NMNLoader

	def _get_nmn_data(self, batch_data):
		keys = ['features', 'question', 'length', 'yesno', 'root_inst', 'find_inst']
		return [ batch_data[k] for k in keys ]

	def _forward(self, batch_data):
		batch_data = cudalize_dict(batch_data, exclude=['find_inst'])
		nmn_data = self._get_nmn_data(batch_data)
		pred = self._model(*nmn_data)
		return dict(
			output = pred,
			label  = batch_data['label'],
			distr  = batch_data['distr']
		)

	def _log_routine(self, mean_loss):
		super(NMNRunner, self)._log_routine(mean_loss)
		self._model.show_times()
