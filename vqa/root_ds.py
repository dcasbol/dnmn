from .vqa_ds import VQADataset
from misc.constants import *
from misc.util import is_yesno
import numpy as np


class VQARootModuleDataset(VQADataset):

	def __init__(self, *args, exclude=None, **kwargs):
		super(VQARootModuleDataset, self).__init__(*args, **kwargs)
		self._hmap_pat = os.path.join(self._root_dir, CACHE_HMAP_FILE)
		self._att_pat = os.path.join(self._root_dir, CACHE_ATT_FILE)
		assert exclude in {None, 'yesno', 'others'}, "Invalid value for 'exclude': {}".format(exclude)

		if exclude is not None:
			yesno_questions = exclude == 'yesno'
			new_id_list = list()
			for qid in self._id_list:
				if is_yesno(self._by_id[qid]['question']) == yesno_questions:
					new_id_list.append(qid)
			self._id_list = new_id_list
			print('Filtered dataset has {} samples'.format(len(self._id_list)))
			assert len(self._id_list) > 0, "No samples were found with exclude = {!r}".format(exclude)

	def _get_hmap(self, datum):
		hmap_fn = self._hmap_pat.format(
			set = datum['input_set'],
			qid = datum['question_id']
		)
		return np.load(hmap_fn)

	def _get_attended(self, datum):
		att_fn = self._att_pat.format(
			set = datum['input_set'],
			qid = datum['question_id']
		)
		return np.load(att_fn)

	def __getitem__(self, i):
		datum = self._get_datum(i)
		labels = np.array(datum['answers'])
		instance = datum['layouts_indices'][0]
		return instance, labels, datum