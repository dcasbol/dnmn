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
parser.add_argument('--save-ext', default='.tiff')
args = parser.parse_args()

cond = -1
if len(args.condition) > 0:
	cond = FIND_INDEX[args.condition]

save_instances = set()
if args.save_instances is not None:
	save_instances = { FIND_INDEX[inst] for inst in args.save_instances }

if args.save_ext[0] != '.':
	args.save_ext = '-' + args.save_ext

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

	inst = instance[0].item()
	if cond > 0 and inst != cond:
		continue
	if len(save_instances) > 0 and inst not in save_instances:
		continue

	save = False
	if inst in save_instances:
		save_instances.remove(inst)
		save = True

	hmap = find[instance](features)
	vis.update(hmap, instance_str, input_set, input_id,
		save_img = save and args.save_imgs,
		save_map = save,
		ext      = args.save_ext
	)
	
	time.sleep(args.wait)

	if len(save_instances) == 0 and args.save_instances is not None:
		break
