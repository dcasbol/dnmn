import time
import torch
from torch.utils.data import DataLoader
from model.clevr_nmn import CLEVRNMN
from clevr import CLEVRDataset
from misc.util import cudalize, program_depth
import argparse
import random
import numpy as np

def get_args():
	parser = argparse.ArgumentParser(description='Train CLEVR')
	parser.add_argument('mode', choices=['modular','classic','classic_andor'])
	parser.add_argument('--depth', type=int)
	parser.add_argument('--cv-learning', action='store_true')
	args = parser.parse_args()
	return args

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

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

batch_size = 64

args = get_args()

dataset = CLEVRDataset(max_prog_depth=args.depth)
loader  = DataLoader(
	dataset,
	batch_size  = batch_size,
	shuffle     = True,
	num_workers = 4,
	collate_fn  = collate_fn
)

valset = CLEVRDataset(
	json_path='/DataSets/CLEVR_v1.0/questions/CLEVR_val_questions.json',
	max_prog_depth = args.depth,
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
	neural_dtypes = args.mode == 'modular',
	force_andor   = args.mode == 'classic_andor'
)
model = cudalize(model)

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

MAX_N_WORSE = 10
n_worse = 0
best_error = 1.0
cv_prog_depth = None
depths_available = None
initialize_sets  = True
if args.cv_learning:
	depths_available = { program_depth(q['program']) for q in dataset._questions }
	depths_available = list(sorted(depths_available))

epoch = -1
while True:
	epoch += 1

	print('Epoch', epoch)

	if args.cv_learning and initialize_sets and len(depths_available) > 0:
		initialize_sets = False
		cv_prog_depth = None if len(depths_available) == 0 else depths_available[0]
		if len(depths_available) > 0:
			depths_available = depths_available[1:]
		trainset = CLEVRDataset(
			max_prog_depth=cv_prog_depth,
			answer_index = dataset.answer_index,
			find_index   = dataset.find_index,
			desc_index   = dataset.desc_index,
			rel_index    = dataset.rel_index
		)
		loader  = DataLoader(
			trainset,
			batch_size  = batch_size,
			shuffle     = True,
			num_workers = 4,
			collate_fn  = collate_fn
		)
		valset = CLEVRDataset(
			json_path='/DataSets/CLEVR_v1.0/questions/CLEVR_val_questions.json',
			max_prog_depth = cv_prog_depth,
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
		if args.cv_learning and len(depths_available) > 0:
			initialize_sets = True
		else:
			print(n_worse, 'epochs without improving. Stop training')
			break

print('Training finished')
print('Best error:', best_error)

