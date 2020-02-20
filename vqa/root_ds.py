import os
from .vqa_ds import VQADataset
from misc.constants import *
from misc.util import is_yesno
import numpy as np


def _load_cache(pat, datum):
	return np.load(pat.format(
		set = datum['input_set'],
		qid = datum['question_id']
	))

class VQARootModuleDataset(VQADataset):

	def __init__(self, *args, exclude=None, prior=False, **kwargs):
		super(VQARootModuleDataset, self).__init__(*args, **kwargs)
		self._prior = prior
		self._hmap_pat = os.path.join(self._root_dir, CACHE_HMAP_FILE)
		self._att_pat  = os.path.join(self._root_dir, CACHE_ATT_FILE)
		self._pri_pat  = os.path.join(self._root_dir, CACHE_QENC_FILE)
		assert exclude in {None, 'yesno', 'others'}, "Invalid value for 'exclude': {}".format(exclude)

		if exclude is not None:
			yesno_questions = exclude != 'yesno'
			new_id_list = list()
			for qid in self._id_list:
				if is_yesno(self._by_id[qid]['question']) == yesno_questions:
					new_id_list.append(qid)
			self._id_list = new_id_list
			print('Filtered dataset has {} samples'.format(len(self._id_list)))
			assert len(self._id_list) > 0, "No samples were found with exclude = {!r}".format(exclude)

	def _get_prior(self, datum):
		return _load_cache(self._pri_pat, datum)

	def _get_hmap(self, datum):
		return _load_cache(self._hmap_pat, datum)

	def _get_attended(self, datum):
		return _load_cache(self._att_pat, datum)

	def __getitem__(self, i):
		datum = self._get_datum(i)
		labels = np.array(datum['answers'])
		instance = datum['layouts_indices'][0]
		return instance, labels, datum
