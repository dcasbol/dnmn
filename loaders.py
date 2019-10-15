from torch.utils.data import DataLoader
from vqa import VQAFindDataset, VQADescribeDataset, VQAMeasureDataset
from vqa import VQAEncoderDataset, VQANMNDataset, VQAGaugeFindDataset
from vqa import encoder_collate_fn, nmn_collate_fn


class BaseLoader(DataLoader):

	def __init__(self, **kwargs):
		ds_keys = {'root_dir', 'set_names', 'start', 'stop', 'metadata'}
		ds_kwargs = { k:v for k,v in kwargs.items() if k in ds_keys }
		dl_kwargs = { k:v for k,v in kwargs.items() if k not in ds_keys }
		dataset = self._dataset(**ds_kwargs)
		self._dataset_len = len(dataset)
		super(BaseLoader, self).__init__(dataset, **dl_kwargs)
	
	def _dataset(self, **kwargs):
		raise NotImplementedError

	@property
	def dataset_len(self):
		return self._dataset_len
	

class EncoderLoader(BaseLoader):
	def __init__(self, **kwargs):
		super(EncoderLoader, self).__init__(collate_fn=encoder_collate_fn, **kwargs)
	def _dataset(self, **kwargs):
		return VQAEncoderDataset(**kwargs)

class NMNLoader(BaseLoader):
	def __init__(self, **kwargs):
		super(NMNLoader, self).__init__(collate_fn=nmn_collate_fn, **kwargs)
	def _dataset(self, **kwargs):
		return VQANMNDataset(**kwargs)

class FindLoader(BaseLoader):
	def _dataset(self, **kwargs):
		return VQAFindDataset(**kwargs)

class DescribeLoader(BaseLoader):
	def _dataset(self, **kwargs):
		return VQADescribeDataset(**kwargs)

class MeasureLoader(BaseLoader):
	def _dataset(self, **kwargs):
		return VQAMeasureDataset(**kwargs)

class GaugeFindLoader(BaseLoader):
	def _dataset(self, **kwargs):
		return VQAGaugeFindDataset(**kwargs)
