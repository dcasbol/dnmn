import time
import torch
from torch.utils.data import DataLoader
from model.clevr_nmn import CLEVRNMN
from clevr import CLEVRDataset
from misc.util import cudalize

def collate_fn(data):
	tensor_data = list()
	targets     = list()
	for datum in data:
		datum['features'] = torch.as_tensor(datum['features'], dtype=torch.float)
		tensor_data.append(datum)
		targets.append(datum['answer'])
	return dict(
		samples = tensor_data,
		answer  = torch.tensor(targets, dtype=torch.long)
	)

batch_size = 512

dataset = CLEVRDataset(max_prog_depth=5)
loader  = DataLoader(
	dataset,
	batch_size  = batch_size,
	shuffle     = True,
	num_workers = 4,
	collate_fn  = collate_fn
)

valset = CLEVRDataset(
	json_path='/DataSets/CLEVR_v1.0/questions/CLEVR_val_questions.json',
	max_prog_depth = 5,
	answer_index = dataset.answer_index,
	find_index   = dataset.find_index,
	desc_index   = dataset.desc_index,
	rel_index    = dataset.rel_index
)
val_loader = DataLoader(
	valset,
	batch_size = batch_size,
	num_workers = 4,
	collate_fn = collate_fn
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

MAX_N_WORSE = 10
n_worse = 0
best_error = 1.0
for epoch in range(500):

	print('Epoch', epoch)

	t0 = time.time()
	for i, batch_data in enumerate(loader):

		output = model(batch_data)
		loss   = output['loss']

		opt.zero_grad()
		loss.backward()
		opt.step()

		if time.time()-t0 > 10:
			t0 = time.time()
			print(int(100*i/len(loader)), '/', loss.item())

	# Validation
	N = 0
	val_error = 0
	model.eval()
	with torch.no_grad():
		for batch_data in val_loader:
			output = model(batch_data)
			ans = output['answer'].max(1).indices
			targets = cudalize(batch_data['answer'])
			err = (ans != targets).float().sum()
			
			B = ans.size(0)
			N += B
			val_error += err.item()
	model.train()
	val_error /= N
	print('Val error:', val_error)
	if val_error < best_error:
		print('Saving checkpoint')
		model.save()
		best_error = val_error
		n_worse = 0
	else:
		n_worse += 1

	if n_worse == MAX_N_WORSE:
		print(n_worse, 'epochs without improving. Stop training')
		break

print('Training finished')

