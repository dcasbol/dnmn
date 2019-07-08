import torch
import torch.nn as nn
from modules import Find, Describe, Measure, QuestionEncoder

class NMN(nn.Module):

	def __init__(self):
		super(NMN, self).__init__()
		self._find = FindModule()
		self._describe = Describe()
		self._measure = Measure()
		self._encoder = QuestionEncoder()

	def forward(self, questions, lengths, ...):

		# It'd be optimal to have a set of instances for find for each img.
		