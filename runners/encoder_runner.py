from .base import Runner
from model import QuestionEncoder
from loaders import EncoderLoader
from misc.util import cudalize
from misc.constants import *

class EncoderRunner(Runner):

	def __init__(self, embedding_size=EMBEDDING_SIZE, hidden_units=HIDDEN_UNITS, **kwargs):
		assert 'modular' not in kwargs, "Encoder training doesn't accept modular flag"
		self._embedding_size = embedding_size
		self._hidden_units   = hidden_units
		super(EncoderRunner, self).__init__(**kwargs)

	def _get_model(self):
		return QuestionEncoder(
			dropout        = self._dropout,
			embedding_size = self._embedding_size,
			hidden_units   = self._hidden_units
		)

	def _get_loader(self, **kwargs):
		return EncoderLoader(**kwargs)

	def _forward(self, batch_data):
		question, length, label = cudalize(*batch_data)
		output = self._model(question, length)
		return dict(
			output = output,
			label  = label
		)