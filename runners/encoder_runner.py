from .base import Runner
from model import QuestionEncoder
from loaders import EncoderLoader
from misc.util import cudalize

class EncoderRunner(Runner):

	def __init__(self, **kwargs):
		assert 'modular' not in kwargs, "Encoder training doesn't accept modular flag"
		super(EncoderRunner, self).__init__(**kwargs)

	def _get_model(self):
		return QuestionEncoder(dropout=self._dropout)

	def _get_loader(self, **kwargs):
		return EncoderLoader(**kwargs)

	def _forward(self, batch_data):
		question, length, label = cudalize(*batch_data)
		output = self._model(question, length)
		return dict(
			output = output,
			label  = label
		)