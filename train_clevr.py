import time
import torch
from torch.utils.data import DataLoader
from model.clevr_nmn import CLEVRNMN
from clevr import CLEVRDataset
from misc.util import seed, cudalize, program_depth
import argparse

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

seed()

batch_size = 64
args = get_args()

dataset = CLEVRDataset(max_prog_depth=args.depth)
index_names = ['answer','find','desc','rel']
index_names = [ n+'_index' for n in index_names ]
indices = { name : getattr(dataset, name) for name in index_names }

def get_loader(val=False, depth=None):
	path = None
	if val:
		path = '/DataSets/CLEVR_v1.0/questions/CLEVR_val_questions.json'
	ds = CLEVRDataset(json_path=path,
		min_prog_depth=depth, max_prog_depth=depth, **indices)
	return DataLoader(ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

if not args.cv_learning:
	loader = get_loader(depth=args.depth)
	val_loader = get_loader(val=True, depth=args.depth)

model = CLEVRNMN(
	neural_dtypes = args.mode == 'modular',
	force_andor   = args.mode == 'classic_andor',
	**indices
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

	if args.cv_learning and initialize_sets:
		initialize_sets = False
		cv_prog_depth = None if len(depths_available) == 0 else depths_available[0]
		if len(depths_available) > 0:
			depths_available = depths_available[1:]
		loader = get_loader(depth=cv_prog_depth)
		val_loader = get_loader(val=True, depth=cv_prog_depth)

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
		if args.cv_learning and cv_prog_depth is not None:
			initialize_sets = True
		else:
			print(n_worse, 'epochs without improving. Stop training')
			break

print('Training finished')
print('Best error:', best_error)

