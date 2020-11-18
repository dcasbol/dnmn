import torch
import torch.nn as nn
from misc.util import cudalize, attend_features
import clevr.clevr_modules as modules

class CLEVRNMN(nn.Module):

	def __init__(self, answer_index, find_index, desc_index, rel_index,
		neural_dtypes=False):
		super().__init__()
		self._ndtypes = neural_dtypes
		self._loss_fn = nn.CrossEntropyLoss(reduction='mean')
		self._find = modules.CLEVRFind(len(find_index), neural_dtypes=neural_dtypes)
		self._describe = modules.CLEVRDescribe(len(desc_index), len(answer_index),
			neural_dtypes = neural_dtypes)
		self._relate = modules.CLEVRRelate(len(rel_index), neural_dtypes=neural_dtypes)
		self._measure = modules.CLEVRMeasure(len(answer_index), neural_dtypes=neural_dtypes)
		self._compare = modules.CLEVRCompare(len(answer_index), neural_dtypes=neural_dtypes)

	def _and(self, mask_A, mask_B):
		if self._ndtypes:
			return torch.min(mask_A, mask_B)
		return mask_A * mask_B

	def _or(self, mask_A, mask_B):
		if self._ndtypes:
			return torch.max(mask_A, mask_B)
		return mask_A + mask_B

	def forward(self, batch_data):

		answer_list = list()

		for datum in batch_data:

			features = cudalize(datum['features'].unsqueeze(0))
			in_mask  = self._find[datum['program'][0]['instance']](in_feats)
			first_result = dict(mask = in_mask)
			results = list()

			for instr in datum['program']:

				inputs = [ results[i] for i in instr['inputs'] ]
				module = instr['module']
				instance = instr.get('instance', None)

				if len(inputs) == 0:
					res = first_result
				elif module == 'find':
					assert len(inputs) == 1
					mask = self._find[instance](features)
					mask = self._and(inputs[0]['mask'], mask)
					res = dict(mask = mask)
				elif module == 'describe':
					assert len(inputs) == 1
					logits = self._describe[instance](inputs[0]['descriptor'])
					res = dict(logits = logits, softmax = logits.softmax(1) )
				elif module == 'relate':
					assert len(inputs) == 1
					mask = self._relate[instance](inputs[0]['descriptor'], features)
					res = dict(mask = mask)
				elif module == 'measure':
					assert len(inputs) == 1
					logits = self._measure[instance](inputs[0]['mask'])
					res = dict(logits = logits, softmax = logits.softmax(1) )
				elif module == 'attend':
					assert len(inputs) == 1
					descriptor = attend_features(features, inputs[0]['mask'])
					res = dict(descriptor = descriptor)
				elif module == 'intersect':
					assert len(inputs) == 2
					mask = self._and(inputs[0]['mask'], inputs[1]['mask'])
					res = dict(mask = mask)
				elif module == 'union':
					assert len(inputs) == 2
					mask = self._or(inputs[0]['mask'], inputs[1]['mask'])
					res = dict(mask = mask)
				elif module == 'compare':
					assert len(inputs) == 2
					key = 'softmax' if self._ndtypes else 'logits'
					logits = self._compare[instance](inputs[0][key], inputs[1][key])
					res = dict(logits = logits, softmax = logits.softmax(1) )
				else:
					raise ValueError('Unknown module {}'.format(module))

				results.append(res)

			answer = results[-1]['logits']
			answer_list.append(answer)

		output = dict(answer = torch.cat(answer_list, dim=0))
		if self.training:
			output['loss'] = self._loss_fn(output['answer'], batch_data['answer'])

		return output