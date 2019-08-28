import json
from glob import glob

prefix = 'describe-qual-ep-'
i0 = len(prefix)

fn_list = glob(prefix+'*.json')
fn_list.sort()

acc_list = list()
epoch_list = list()
max_epoch_list = list()

for fn in fn_list:
	with open(fn) as fd:
		data = json.load(fd)
	max_acc = max(data['top_1'])
	if data['top_1'][-1] < max_acc:
		acc_list.append(max_acc)
		epoch_list.append(int(fn[i0:i0+2]))
		max_epoch_list.append(data['epoch'][-4])

result = dict(
	epoch = epoch_list,
	top_1 = acc_list,
	max_epoch = max_epoch_list
)

with open('find-qual_log.json') as fd:
	wvar = json.load(fd)['wvar']

wvar_list = [ wvar[epoch] for epoch in epoch_list ]
result['wvar'] = wvar_list

with open('test.json', 'w') as fd:
	json.dump(result, fd)
