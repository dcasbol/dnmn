from .base_module import BaseModule


class InstanceModule(BaseModule):
	"""Module with [] overloaded to follow nomenclature as in paper:
	Find[inst](features)"""

	def __init__(self, **kwargs):
		super(InstanceModule, self).__init__(**kwargs)
		self._instance = None

	def __getitem__(self, instance):
		assert self._instance is None
		self._instance = instance
		return self

	def _get_instance(self):
		inst = self._instance
		assert inst is not None, "Can't call module without instance"
		self._instance = None
		return inst