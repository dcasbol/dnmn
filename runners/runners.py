import torch
from modules import Find, Describe, Measure, QuestionEncoder
from model import NMN
from runners.base import Runner
from misc.visualization import MapVisualizer
from misc.util import cudalize, cudalize_dict, to_tens, DEVICE
from loaders import EncoderLoader, FindLoader, DescribeLoader, MeasureLoader, NMNLoader


class FindRunner(Runner):

	def __init__(self, competition='softmax', visualize=0, dropout=0, **kwargs):
		self._model   = Find(competition=competition, dropout=dropout)
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
			label     = instance,
			hmap      = output,
			label_str = label_str,
			input_set = input_set,
			input_id  = input_id,
			features  = features,
		)
		return self._result

	def _preview(self, mean_loss):
		super(FindRunner, self)._preview(mean_loss)
		if self._visualize > 0:
			keys   = ['hmap', 'label_str', 'input_set', 'input_id']
			values = [ self._result[k] for k in keys ]
			self._vis.update(*values)

	def _validation_routine(self):
		if not self._validate: return
		N = wvar = 0

		self._model.eval()
		with torch.no_grad():
			for batch_data in self._val_loader:
				result = self._forward(batch_data)
				att = self._attend(result['features'], result['hmap'])
				wvar += self._weighted_var()

		self._model.train()
		
		self._logger.log(
			top_1    = top1/N,
			in_set   = inset/N,
			weighted = wacc/N
		)
		self._logger.print(exclude=['raw_time', 'time', 'epoch', 'loss'])

	def _attend(self, features, hmap):
		B,C,H,W = features.size()
		features = features.view(B,C,-1)
		hmap = hmap.view(B,1,-1)
		total = hmap.sum(2)
		attended = (hmap*features).sum(2) / (hmap.sum(2) + 1e-10)
		return dict(
			features_flat = features,
			hmap_flat = hmap,
			total = total,
			attended = attended
		)

	def _weighted_var(self, features, hmap, attended, total):
		B, C = attended.size()[:2]
		attended = attended.view(B,C,1)
		var = (features-attended).pow(2)
		wvar = (var*hmap).sum(2) / (total + 1e-10)
		return wvar.mean()

class DescribeRunner(Runner):

	def __init__(self, dropout=0, **kwargs):
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

class UncachedRunner(Runner):

	def __init__(self, find_pt='find.pt', **kwargs):
		super(UncachedRunner, self).__init__(**kwargs)
		self._find = Find(competition=None)
		self._find.load_state_dict(torch.load(find_pt, map_location='cpu'))
		self._find = cudalize(self._find)
		self._find.eval()

	def _loader_class(self):
		return NMNLoader

	def _forward(self, batch_data):
		features  = cudalize(batch_data['features'])
		root_inst = cudalize(batch_data['root_inst'])
		find_inst = [ to_tens(inst, 'long', DEVICE) for inst in batch_data['find_inst'] ]

		with torch.no_grad():
			features_list = features.unsqueeze(1).unbind(0)
			maps = list()
			for f, inst in zip(features_list, find_inst):
				f = f.expand(len(inst), -1, -1, -1)
				m = self._find[inst](f).prod(0, keepdim=True)
				maps.append(m)
			maps = torch.cat(maps)

		return dict(
			output = self._model[root_inst](maps, features),
			label  = cudalize(batch_data['label']),
			distr  = cudalize(batch_data['distr'])
		)

class DescribeRunnerUncached(UncachedRunner):

	def __init__(self, dropout=0, **kwargs):
		self._model = Describe(dropout=dropout)
		super(DescribeRunnerUncached, self).__init__(**kwargs)

class MeasureRunnerUncached(UncachedRunner):

	def __init__(self, dropout=0, **kwargs):
		self._model = Measure(dropout=dropout)
		super(MeasureRunnerUncached, self).__init__(**kwargs)

class MeasureRunner(Runner):

	def __init__(self, dropout=0, **kwargs):
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

	def __init__(self, dropout=0, embed_size=None, **kwargs):
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

	def __init__(self, dropout=0, **kwargs):
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
