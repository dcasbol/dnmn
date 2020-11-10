from .root_ds import VQARootModuleDataset
from misc.util import is_yesno
import numpy as np
from misc.indices import FIND_INDEX


class VQAGaugeFindDataset(VQARootModuleDataset):

	def __init__(self, *args, metadata=False, inst=False, **kwargs):
		super(VQAGaugeFindDataset, self).__init__(*args, **kwargs)
		self._metadata = metadata
		self._inst     = inst

	def __getitem__(self, i):
		datum    = self._get_datum(i)
		features = self._get_features(datum)

		target_list = list()
		for name, index in zip(datum['layouts_names'], datum['layouts_indices']):
			if name != 'find': continue
			target_list.append(index)

		n = len(target_list)
		if n == 2:
			target_1, target_2 = target_list
		else:
			target_1 = target_list[0]
			target_2 = 0

		yesno  = is_yesno(datum['question'])
		labels = np.array(datum['answers'])
		
		output = (features, target_1, target_2, yesno, labels)
		if self._inst:
			output += (datum['layouts_indices'][0],)
		if self._prior:
			output += (self._get_prior(datum),)
		if self._metadata:
			target_str = FIND_INDEX.get(target_1)
			if target_2 > 0:
				target_str += ' AND ' + FIND_INDEX.get(target_2)
			output += (target_str, datum['input_set'], datum['input_id'])

		return output
