from .root_ds import VQARootModuleDataset

class VQAMeasureDataset(VQARootModuleDataset):
	def __init__(self, *args, **kwargs):
		super(VQAMeasureDataset, self).__init__(*args, **kwargs, exclude='others')

	def __getitem__(self, i):
		instance, labels, datum = super(VQAMeasureDataset, self).__getitem__(i)
		hmap = self._get_hmap(datum)
		if self._prior:
			return hmap, instance, labels, self._get_prior(datum)
		return hmap, instance, labels