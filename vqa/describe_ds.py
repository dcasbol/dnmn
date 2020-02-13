from .root_ds import VQARootModuleDataset

class VQADescribeDataset(VQARootModuleDataset):
	def __init__(self, *args, **kwargs):
		super(VQADescribeDataset, self).__init__(*args, **kwargs, exclude='yesno')

	def __getitem__(self, i):
		instance, labels, datum = super(VQADescribeDataset, self).__getitem__(i)
		att = self._get_attended(datum)
		if self._prior:
			return att, instance, labels, self._get_prior(datum)
		return att, instance, labels