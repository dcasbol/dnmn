import torch
import torch.nn.functional as F
from misc.util import cudalize, DEVICE, Chronometer
from modules import Find, Describe, Measure, QuestionEncoder
from misc.indices import ANSWER_INDEX

class NMN(torch.nn.Module):

	def __init__(self, dropout=False):
		super(NMN, self).__init__()
		self._find = Find(competition=None)
		self._describe = Describe(normalize_attention=False)
		self._measure = Measure(dropout=dropout)
		self._encoder = QuestionEncoder(dropout=dropout)

		self._dropout = {
			False : lambda x: x,
			True  : lambda x: F.dropout(x, p=0.5, training=self.training)
		}[dropout]
		self._dropout2d = {
			False : lambda x: x,
			True  : lambda x: F.dropout2d(x, p=0.5, training=self.training)
		}[dropout]
		self._chr = { k: Chronometer() for k in ['do', 'inst', 'maps', 'empty', 'yesno', 'others', 'qenc'] }

	def forward(self, features, question, length, yesno, root_inst, find_inst):

		# this is the equivalent to dropout over 1x1 kernel
		# Drop here to drop the same features for all modules
		self._chr['do'].start()
		features = self._dropout2d(self._dropout(features))
		features_list = features.unsqueeze(1).unbind(0)
		self._chr['do'].stop()

		self._chr['inst'].start()
		find_inst = [ torch.tensor(inst, dtype=torch.long, device=DEVICE) for inst in find_inst ]
		self._chr['inst'].stop()

		self._chr['maps'].start()
		maps = list()
		for f, inst in zip(features_list, find_inst):
			f = f.expand(len(inst), -1, -1, -1)
			m = self._find[inst](f).prod(0, keepdim=True)
			maps.append(m)
		maps = torch.cat(maps)
		self._chr['maps'].stop()

		self._chr['empty'].start()
		root_pred = torch.empty(yesno.size(0), len(ANSWER_INDEX), device=DEVICE)
		self._chr['empty'].stop()

		self._chr['yesno'].start()
		yesno_maps = maps[yesno]
		if yesno_maps.size(0) > 0:
			yesno_inst = root_inst[yesno]
			yesno_ans = self._measure[yesno_inst](yesno_maps)
			root_pred[yesno] = yesno_ans
		self._chr['yesno'].stop()

		self._chr['others'].start()
		other = ~yesno
		other_maps = maps[other]
		if other_maps.size(0) > 0:
			other_inst = root_inst[other]
			other_fts  = features[other]
			other_ans  = self._describe[other_inst](other_maps, other_fts)
			root_pred[other] = other_ans
		self._chr['others'].stop()

		self._chr['qenc'].start()
		root_pred = root_pred.softmax(1)
		enc_pred = self._encoder(question, length).softmax(1)
		self._chr['qenc'].stop()

		return (root_pred*enc_pred + 1e-30).sqrt()

	def load(self, filename):
		self.load_state_dict(torch.load(filename, map_location='cpu'))

	def save(self, filename):
		torch.save(self.state_dict(), filename)

	def load_module(self, module_name, filename):
		assert module_name in {'find', 'describe', 'measure', 'encoder'}
		name = '_'+module_name
		module = getattr(self, name)
		module.load_state_dict(torch.load(filename, map_location='cpu'))
		setattr(self, name, cudalize(module))

	def save_module(self, module_name, filename):
		assert module_name in {'find', 'describe', 'measure', 'encoder'}
		module = getattr(self, '_'+module_name)
		torch.save(module.state_dict(), filename)

	def show_times(self):
		for k, c in self._chr.items():
			print(c, c.read())