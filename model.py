import torch
from misc.util import cudalize
from modules import Find, Describe, Measure, QuestionEncoder
from misc.indices import ANSWER_INDEX

class NMN(torch.nn.Module):

	def __init__(self):
		super(NMN, self).__init__()
		self._find = Find(competition=None)
		self._describe = Describe()
		self._measure = Measure()
		self._encoder = QuestionEncoder()

	def forward(self, features, question, length, yesno, root_inst, find_inst):

		features_list = features.unsqueeze(1).unbind(0)

		maps = list()
		for f, inst in zip(features_list, find_inst):
			f = f.expand(len(inst), -1, -1, -1)
			inst = cudalize(torch.tensor(inst, dtype=torch.long))
			m = self._find[inst](f).prod(0, keepdim=True)
			maps.append(m)
		maps = torch.cat(maps)

		root_pred = cudalize(torch.empty(yesno.size(0), len(ANSWER_INDEX)))

		yesno_maps = maps[yesno]
		if yesno_maps.size(0) > 0:
			yesno_inst = root_inst[yesno]
			yesno_ans = self._measure[yesno_inst](yesno_maps)
			root_pred[yesno] = yesno_ans

		other = ~yesno
		other_maps = maps[other]
		if other_maps.size(0) > 0:
			other_inst = root_inst[other]
			other_fts  = features[other]
			other_ans  = self._describe[other_inst](other_maps, other_fts)
			root_pred[other] = other_ans

		root_pred = root_pred.softmax(1)
		enc_pred = self._encoder(question, length).softmax(1)

		return (root_pred*enc_pred).sqrt()

	def load(self, module_name, filename):
		assert module_name in {'find', 'describe', 'measure', 'encoder'}
		name = '_'+module_name
		module = getattr(self, name)
		module.load_state_dict(torch.load(filename, map_location='cpu'))
		setattr(self, name, cudalize(module))

	def save(self, module_name, filename):
		assert module_name in {'find', 'describe', 'measure', 'encoder'}
		module = getattr(self, '_'+module_name)
		torch.save(module.state_dict(), filename)
