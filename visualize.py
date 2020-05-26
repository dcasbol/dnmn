import argparse
import torch
import time
from torch.utils.data import DataLoader
from vqa import VQAFindDataset
from model import Find, NMN
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
parser.add_argument('--from-nmn', action='store_true')
parser.add_argument('--save-instances', nargs='*')
parser.add_argument('--save-imgs', action='store_true')
args = parser.parse_args()

cond = -1
if len(args.condition) > 0:
	cond = FIND_INDEX[args.condition]
save_instances = { FIND_INDEX[inst] for inst in args.save_instances }

findset = VQAFindDataset(metadata=True)
loader = DataLoader(findset, batch_size=1, shuffle=args.shuffle)

if args.from_nmn:
	nmn = NMN(modular=args.modular)
	nmn.load(args.ptfile)
	find = nmn._find
else:
	find = Find(modular=args.modular)
	find.load(args.ptfile)
find.eval()

vis = MapVisualizer(1, vmin = 0, vmax = 1 if args.modular else None)

for features, instance, instance_str, input_set, input_id in loader:

	if cond > 0 and instance[0].item() != cond:
		continue

	save = False
	if instance in save_instances:
		save_instances.remove(instance)
		save = True

	hmap = find[instance](features)
	vis.update(hmap, instance_str, input_set, input_id,
		save_img = save and args.save_imgs,
		save_map = save
	)
	
	time.sleep(args.wait)

	if len(save_instances) == 0 and len(args.save_instances) > 0:
		break
