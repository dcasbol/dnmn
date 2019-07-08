from vqa import VQANMNDataset, nmn_collate_fn
from torch.utils.data import DataLoader
from misc.indices import ANSWER_INDEX, DESC_INDEX, QUESTION_INDEX
from collections import defaultdict


print(len(DESC_INDEX))
for i in DESC_INDEX:
	print(i, DESC_INDEX[i])
quit()

dataset = VQANMNDataset()
loader = DataLoader(dataset,
	batch_size = 4,
	collate_fn = nmn_collate_fn
)

for batch in loader:

	names = ['padded', 'lengths', 'yesno', 'features', 'indices', 'label', 'distr']
	for n, v in zip(names, batch):
		if type(v) == tuple:
			print(n, [len(vi) for vi in v])
		else:
			print(n, v.size())
	quit()