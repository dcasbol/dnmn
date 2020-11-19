import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.constants import *


class CLEVRBaseModule(nn.Module):

	def __init__(self, name=None, saving_dir='./'):
		super().__init__()
		self._name = name
		self._saving_dir = saving_dir
		self._instance = None

	def __getitem__(self, instance):
		assert self._instance is None
		self._instance = instance
		return self

	def _get_instance(self):
		inst = self._instance
		assert inst is not None
		self._instance = None
		return inst

	def save(self, filename=None):
		alt_filename = self._name + '.pt'
		filename = filename or alt_filename
		assert filename is not None
		path = os.path.join(self._saving_dir, filename)
		torch.save(self.state_dict(), path)

	def load(self, filename=None):
		filename = filename or self._name
		assert filename is not None
		path = os.path.join(self._saving_dir, filename)
		self.load_state_dict(torch.load(path, map_location='cpu'))


class CLEVRFind(CLEVRBaseModule):

	def __init__(self, num_instances, neural_dtypes=False):
		suffix = 'modular' if neural_dtypes else 'classic'
		super().__init__(name='find_'+suffix)
		self._instances = nn.ModuleList([
			nn.Conv2d(IMG_DEPTH+2, 1, 1)
			for _ in range(num_instances)
		])
		self._act = nn.Sigmoid() if neural_dtypes else nn.ReLU()

	def forward(self, features):
		inst = self._get_instance()
		return self._act(self._instances[inst](features))


class CLEVRDescribe(CLEVRBaseModule):

	def __init__(self, num_instances, num_answers, neural_dtypes=False):
		suffix = 'modular' if neural_dtypes else 'classic'
		super().__init__(name='describe_'+suffix)
		self._instances = nn.ModuleList([
			nn.Linear(IMG_DEPTH+2, num_answers)
			for _ in range(num_instances)
		])

	def forward(self, descriptor):
		inst = self._get_instance()
		return self._instances[inst](descriptor)


class CLEVRRelate(CLEVRBaseModule):

	def __init__(self, num_instances, neural_dtypes=False):
		suffix = 'modular' if neural_dtypes else 'classic'
		super().__init__(name='relate_'+suffix)
		self._instances = nn.ModuleList([
			nn.Linear(IMG_DEPTH+2, IMG_DEPTH+2)
			for _ in range(num_instances)
		])
		self._act = nn.Sigmoid() if neural_dtypes else nn.ReLU()

	def forward(self, descriptor, features):
		inst = self._get_instance()
		w = self._instances[inst](descriptor)
		w = w.view(1, IMG_DEPTH+2, 1, 1)
		return self._act(F.conv2d(features, w))


class CLEVRMeasure(CLEVRBaseModule):

	def __init__(self, num_answers, neural_dtypes=False):
		suffix = 'modular' if neural_dtypes else 'classic'
		super().__init__(name='measure_'+suffix)
		self._instances = nn.ModuleList([
			nn.Sequential(
				nn.Linear(10*15, 100),
				nn.ReLU(),
				nn.Linear(100, num_answers)
			)
			for _ in range(2)
		])

	def forward(self, mask):
		inst = self._get_instance()
		return self._instances[inst](mask.view(-1,10*15))


class CLEVRCompare(CLEVRBaseModule):

	def __init__(self, num_answers, neural_dtypes=False):
		suffix = 'modular' if neural_dtypes else 'classic'
		super().__init__(name='compare_'+suffix)
		self._instances = nn.ModuleList([
			nn.Linear(num_answers, num_answers)
			for _ in range(3)
		])

	def forward(self, pred_A, pred_B):
		inst = self._get_instance()
		comp = pred_A - pred_B
		if inst == 0:
			comp = comp.abs()
		return self._instances[inst](comp)