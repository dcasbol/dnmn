import torch
import torch.nn as nn
from functools import reduce
from misc.util import cudalize
from modules import Find, Describe, Measure, QuestionEncoder

class NMN(nn.Module):

	def __init__(self):
		super(NMN, self).__init__()
		self._find = FindModule(competition=None)
		self._describe = Describe()
		self._measure = Measure()
		self._encoder = QuestionEncoder()

	def forward(self, question, length, yesno, features, instances):

		# Expand features that need +1 find op.
		features = features.unsqueeze(1).unbind(0)

		maps = list()
		for f, inst in zip(features, instances):
			f = f.expand(len(inst), -1, -1, -1)
			inst = cudalize(torch.tensor(inst, dtype=torch.long))
			m = self._find(f, inst)
			maps.append(m)
		maps = torch.cat(maps)

		# Root nodes
		yesno_maps = maps[yesno]
		yesno_inst = 
		yesno_ans = self._measure()
		others = ~yesno
		other_maps = maps[others]
		other_fts  = features[others]

		enc = self._encoder(question, length)
