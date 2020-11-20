import os
import sys
import time
import json
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

batch_size = 64
prog_depth = int(sys.argv[1])
suffix     = sys.argv[2]

dataset = CLEVRDataset(max_prog_depth=5)
test_dataset = CLEVRDataset(
	min_prog_depth=prog_depth,
	max_prog_depth=prog_depth,
	answer_index = dataset.answer_index,
	find_index   = dataset.find_index,
	desc_index   = dataset.desc_index,
	rel_index    = dataset.rel_index
)
if len(test_dataset) == 0:
	print('No programs found with depth', prog_depth)
	quit()

loader  = DataLoader(
	test_dataset,
	batch_size  = batch_size,
	num_workers = 4,
	collate_fn  = collate_fn
)
model = CLEVRNMN(
	answer_index = dataset.answer_index,
	find_index   = dataset.find_index,
	desc_index   = dataset.desc_index,
	rel_index    = dataset.rel_index,
	neural_dtypes = suffix == 'modular',
	force_andor   = suffix == 'classic_andor'
)
model.load()

N = 0
test_error = 0
model.eval()
with torch.no_grad():
	for i, batch_data in enumerate(loader):
		output = model(batch_data)
		ans = output['answer'].max(1).indices
		targets = cudalize(batch_data['answer'])
		err = (ans != targets).float().sum()

		N += ans.size(0)
		test_error += err.item()
		if i%100 == 0:
			print('\r{:3d}%'.format(int(100*i/len(loader))), end='')
test_error /= N

results_path = 'clevr_results.json'
res = dict()
if os.path.exists(results_path):
	with open(results_path) as fd:
		res = json.load(fd)
if suffix not in res:
	res[suffix] = dict()
res[suffix][prog_depth] = [test_error, N]
with open(results_path,'w') as fd:
	json.dump(res, fd)
