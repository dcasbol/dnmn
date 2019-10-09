import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='Display benchmark data')
parser.add_argument('jsonpath', type=str)
parser.add_argument('xkey', type=str, help='Key for x values')
parser.add_argument('ykeys', type=str, nargs='*')
parser.add_argument('--title', type=str)
parser.add_argument('--xlabel', type=str)
parser.add_argument('--ylabel', type=str)
args = parser.parse_args()

with open(args.jsonpath) as fd:
	data = json.load(fd)

# Some data is generated on-the-fly
if 'anti' in data.keys():
	for suf in ['', '_train']:
		acc_pred = data['top_1'+suf]
		acc_anti = data['anti'+suf]
		vals = [ ap-aa for ap, aa in zip(acc_pred, acc_anti) ]
		data['acc'+suf] = vals

if args.xkey not in data.keys() or len(args.ykeys) == 0:
	print('Keys:', list(data.keys()))
	quit()

xvals = data[args.xkey]
del data[args.xkey]

plt.figure()

if args.title is not None:
	plt.title(args.title)
if args.xlabel is not None:
	plt.xlabel(args.xlabel)
if args.ylabel is not None:
	plt.ylabel(args.ylabel)

for key in args.ykeys:
	plt.plot(xvals, data[key], label=key)

plt.legend()
plt.show()
