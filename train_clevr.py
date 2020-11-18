import torch
from torch.utils.data import DataLoader
from model import CLEVRNMN
from clevr import CLEVRDataset
from misc.util import cudalize

dataset = CLEVRDataset(max_prog_depth=5)
loader  = DataLoader(
	dataset,
	batch_size  = 32,
	shuffle     = True,
	num_workers = 4
)

model = CLEVRNMN(
	answer_index = dataset.answer_index,
	find_index   = dataset.find_index,
	desc_index   = dataset.desc_index,
	rel_index    = dataset.rel_index,
	neural_dtypes = True
)
model = cudalize(model)

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(1):

	print('Epoch', epoch)

	for i, batch_data in enumerate(loader):

		output = model(batch_data)
		loss   = output['loss']

		opt.zero_grad()
		loss.backward()
		opt.step()

		if i%10 == 0:
			print(i/len(loader), '/', loss.item())
