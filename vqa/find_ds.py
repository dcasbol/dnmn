from .vqa_ds import VQADataset
from misc.util import majority_label
from misc.indices import FIND_INDEX, NEG_ANSWERS


class VQAFindDataset(VQADataset):

	def __init__(self, *args, filter_data=True, metadata=False, 
		filtering='majority', **kwargs):
		super(VQAFindDataset, self).__init__(*args, **kwargs)
		self._metadata = metadata

		assert filtering in ['majority', 'all', 'any'], "Invalid filtering mode {!r}".format(filtering)
		filter_op = dict(
			majority = lambda ans: majority_label(ans) in NEG_ANSWERS,
			all      = lambda ans: set(ans).issubset(NEG_ANSWERS),
			any      = lambda ans: len(set(ans).intersection(NEG_ANSWERS)) > 0
		)[filtering]

		self._imap = list()
		self._tmap = list()
		n_filtered = n_included = 0
		for i, qid in enumerate(self._id_list):

			q = self._by_id[qid]
			if filter_data and filter_op(q['answers']):
				n_filtered += 1
				continue

			lnames = q['layouts_names']
			lindex = q['layouts_indices']
			for j, (name, idx) in enumerate(zip(lnames, lindex)):
				if name != 'find': continue
				n_included += 1
				self._imap.append(i)
				self._tmap.append(j)

		print(n_filtered, 'filtered out,', n_included, 'included')

	def __len__(self):
		return len(self._imap)

	def __getitem__(self, i):
		datum    = self._get_datum(self._imap[i])
		features = self._get_features(datum)

		assert len(datum['parses']) == 1, 'Encountered item ({}) with +1 parses: {}'.format(i, datum['parses'])
		target = datum['layouts_indices']
		target = target[self._tmap[i]] if len(self._tmap) > 0 else target[-1]
		
		output = (features, target)
		if self._metadata:
			target_str = FIND_INDEX.get(target)
			output += (target_str, datum['input_set'], datum['input_id'])

		return output