import os
import torch
from glob import glob
from loaders import FindLoader
from modules import Find
from misc.constants import *
from misc.util import cudalize, Logger

def weighted_var_masked(features, hmap, attended, total):
	B, C = attended.size()[:2]
	attended = attended.view(B,C,1)
	masked = (features*hmap).sqrt()
	var = (masked-attended).pow(2)
	wvar = (var*hmap).sum(2) / (total + 1e-10)
	return wvar.mean()

def common_features(features, hmap):
	B,C,H,W = features.size()
	masks = (hmap > 0).view(B,-1).unbind(0)
	features_list = features.view(B,C,-1).unbind(0)
	selections = [ f[:,m] for f, m in zip(features_list, masks) ]
	present    = [ (s > 0).float() for s in selections ]
	n_common = [ p.prod(1).sum() for p in present ]
	n_common = torch.as_tensor(n_common).mean()
	weighted = [ p.mean(1).sum() for p in present ]
	weighted = torch.as_tensor(weighted).mean()
	return n_common.item(), weighted.item()

def filler(features, hmap):
	B,C,H,W = features.size()
	features_flat = features.view(B,C,H*W)
	indices = features_flat.max(2)[1]
	ih = indices // W
	iw = indices % W
	feature_presence = features > 0
	idx_C = torch.arange(C)
	masks = [ _rec_filler(idx_B, feature_presence[i], ih[i], iw[i]) for i in range(B) ]
	masks = [ _set_to_mask(H, W, m) for m in masks ]
	return torch.cat(masks)

def _rec_filler(idx_C, presence, ih, iw, sel=set(), ref=None):

	new_ref = presence[idx_C, ih, iw]

	if ref is None:
		ref = new_ref
		sel.add((ih,iw))
	elif (ref == new_ref).all():
		ref.add((ih,iw))

	if ih > 0:
		_rec_filler(idx_C, presence, ih-1, iw)
	if ih < presence.size(1)-1:
		_rec_filler(idx_C, presence, ih+1, iw)

	if iw > 0:
		_rec_filler(idx_C, presence, ih, iw-1)
	if iw < presence.size(2)-1:
		_rec_filler(idx_C, presence, ih, iw+1)

	return ref

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
		batch_size = VAL_BATCH_SIZE,
		shuffle    = False
	)

	find = Find(competition=None)
	logger = Logger()

	prefix = 'find-qual-ep'
	i0 = len(prefix)

	pattern = prefix + '*.pt'
	fn_list = glob(pattern)
	fn_list.sort()

	for fn in fn_list:
		print('Applying to {!r}'.format(fn))

		epoch = int(os.path.basename(fn)[i0:i0+2])

		find.load_state_dict(torch.load(fn, map_location='cpu'))
		find = cudalize(find)

		N = n_common_total = weighted_total = 0

		with torch.no_grad():
			for batch_data in val_loader:
				result = run_find(find, batch_data, False)
				n_common, weighted = common_features(result['features'], result['hmap'])
				B = result['features'].size(0)
				N += B
				n_common_total += n_common * B
				weighted_total += weighted * B

		logger.log(
			epoch = epoch,
			n_common = n_common_total/N,
			weighted = weighted_total/N
		)

		logger.save('measure.json')
