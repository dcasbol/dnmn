import os
import torch
from glob import glob
from loaders import FindLoader
from modules import Find
from misc.constants import *
from misc.util import cudalize, Logger
from misc.visualization import MapVisualizer
import time

def weighted_var_masked(features, hmap, attended, total):
	B, C = attended.size()[:2]
	attended = attended.view(B,C,1)
	masked = (features*hmap).sqrt()
	var = (masked-attended).pow(2)
	wvar = (var*hmap).sum(2) / (total + 1e-10)
	return wvar.mean()

def common_features(features, hmap):
	B,C,H,W = features.size()
	masks = (hmap > 0).view(B,1,-1).unbind(0)
	features_list = features.view(B,C,-1).unbind(0)
	selections = [ f[m] for f, m in zip(features_list, masks) ]
	present    = [ (s > 0).float() for s in selections ]
	n_common = [ p.prod(1).sum() for p in present ]
	n_common = torch.cat(n_common).mean()
	weighted = [ p.mean(1).sum() for p in present ]
	weighted = torch.cat(weighted).mean()
	return n_common.item(), weighted.item()

def filler(features, hmap):
	B,C,H,W = features.size()
	features_flat = features.view(B,C,H*W)
	hmap_flat     = hmap.view(B,H*W)
	indices = hmap_flat.max(1)[1]
	ih = indices // W
	iw = indices % W
	idx_C = torch.arange(C)
	sel = {(ih, iw)}
	masks = [ _rec_filler(idx_C, features[i], features[i,idx_C,ih,iw], 0, 0, sel) for i in range(B) ]
	masks = [ _set_to_mask(H, W, m) for m in masks ]
	masks = torch.cat(masks).view(B,1,H,W)

	idx_B = torch.arange(B)
	refs = features[idx_B,idx_C,ih,iw].view(B,C,1,1)
	softmask = (features*refs).sum(1, keepdim=True)
	return masks, softmask

def _rec_filler(idx_C, features, ref_vec, ih, iw, sel=set()):

	if (ih, iw) in sel: return

	new_ref = features[idx_C, ih, iw]

	if torch.matmul(ref_vec, new_ref) / (torch.norm(ref_vec)*torch.norm(new_ref)) > 0.3:
		sel.add((ih,iw))

	H,W = features.size()[1:3]

	if ih >= iw and ih < H-1:
		_rec_filler(idx_C, features, ref_vec, ih+1, iw, sel)

	if iw >= ih and iw < W-1:
		_rec_filler(idx_C, features, ref_vec, ih, iw+1, sel)

	if ih == iw and ih < H-1 and iw < W-1:
		_rec_filler(idx_C, features, ref_vec, ih+1, iw+1, sel)

	return sel

def _set_to_mask(H, W, coord_set):
	mask = torch.zeros(1,H,W, dtype=torch.uint8)
	for ih, iw in coord_set:
		mask[0,ih,iw] = 1
	return mask

def run_find(module, batch_data, metadata):
	if metadata:
		features, instance, label_str, input_set, input_id = batch_data
	else:
		features, instance = batch_data
	features, instance = cudalize(features, instance)
	output = module[instance](features)
	result = dict(
		instance  = instance,
		features  = features,
		output    = output,
		hmap      = output
	)
	if metadata:
		result['label_str'] = label_str
		result['input_set'] = input_set
		result['input_id']  = input_id
	return result

if __name__ == '__main__':

	val_loader = FindLoader(
		set_names  = 'val2014',
		stop       = 0.2,
		batch_size = 1,
		shuffle    = False,
		metadata   = True
	)

	find = Find(competition=None)
	find.eval()
	logger = Logger()

	vis = MapVisualizer(1)

	prefix = 'find-qual-ep'
	i0 = len(prefix)

	pattern = prefix + '*.pt'
	fn_list = glob(pattern)
	fn_list.sort()

	for fn in fn_list:

		epoch = int(os.path.basename(fn)[i0:i0+2])

		find.load_state_dict(torch.load(fn, map_location='cpu'))
		find = cudalize(find)

		N = n_common_total = weighted_total = 0

		with torch.no_grad():
			for batch_data in val_loader:
				result = run_find(find, batch_data, True)
				masks, softmask = filler(result['features'], result['hmap'])
				hmap = result['hmap']
				hmap = hmap / (hmap.max() + 1e-10)
				softmask = softmask / (softmask.max() + 1e-10)
				mean = (softmask + hmap) / 2.
				label_str, input_set, input_id = [ result[k] for k in ['label_str', 'input_set', 'input_id'] ]
				cmap_list = ['viridis', 'plasma', 'hot']
				name_list = [ ' (%s)' % name for name in ['hmap', 'cos-sim', 'mean'] ]
				for m, cmap, name in zip([hmap, softmask, mean], cmap_list, name_list):
					vis._cmap = cmap
					vis.update(m, [label_str[0]+name], input_set, input_id)
					time.sleep(1)