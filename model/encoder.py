import torch
import torch.nn as nn
from .base_module import BaseModule
from misc.constants import *
from misc.indices import QUESTION_INDEX, ANSWER_INDEX


class QuestionEncoder(BaseModule):
	"""Single-layer LSTM with 1024 units. The question modeling component predicts
	a distribution over the set of answers."""

	NAME = 'encoder'

	def __init__(self, dropout=0):
		super(QuestionEncoder, self).__init__(dropout=dropout)
		self._wemb = nn.Embedding(len(QUESTION_INDEX), EMBEDDING_SIZE)
		self._lstm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_UNITS)
		self._final = nn.Linear(HIDDEN_UNITS, len(ANSWER_INDEX))

	def forward(self, question, length):
		B = length.size(0)
		embed = self._wemb(question)
		hidden = self._lstm(embed)[0][length-1, torch.arange(B)]
		return self._final(self._dropout(hidden))