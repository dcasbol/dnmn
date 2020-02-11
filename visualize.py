import argparse
import torch
import time
from torch.utils.data import DataLoader
from vqa import VQAFindDataset
from modules import Find
from misc.indices import FIND_INDEX
from misc.constants import *
from misc.visualization import MapVisualizer

parser = argparse.ArgumentParser(description='Visualize Find Module')
parser.add_argument('ptfile', type=str,
	help='Path to .pt file storing module weights')
parser.add_argument('condition', type=str, nargs='?', default="",
	help='Condition visualizations on class.')
parser.add_argument('--wait', type=float, default=1.)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--dataset', choices=['train2014', 'val2014'], default='val2014')
parser.add_argument('--modular', action='store_true')
args = parser.parse_args()

cond = -1
if len(args.condition) > 0:
	cond = FIND_INDEX[args.condition]

findset = VQAFindDataset(metadata=True)
loader = DataLoader(findset, batch_size=1, shuffle=args.shuffle)

find = Find(modular=args.modular)
find.load_state_dict(torch.load(args.ptfile, map_location='cpu'))
find.eval()

vis = MapVisualizer(1)

for features, instance, instance_str, input_set, input_id in loader:

	if cond > 0 and instance[0].item() != cond:
		continue

	hmap = find[instance](features)
	vis.update(hmap, instance_str, input_set, input_id)
	
	time.sleep(args.wait)
