import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

def _display_tangent(vals):
	vtg = vals[-10]
	for i in range(9):
		vtg = 0.9*vtg + 0.1*vals[-9+i]
	return vtg

parser = argparse.ArgumentParser(description='Display benchmark data')
parser.add_argument('jsonpath', type=str)
args = parser.parse_args()

with open(args.jsonpath) as fd:
	meta, data = json.load(fd)

plt.figure()

plt.title(meta['title'])
plt.xlabel(meta['xlabel'])
plt.ylabel(meta['ylabel'])

keys = list()
y = list()
for k, vals in data.items():
	keys.append(k)
	y.append(_display_tangent(vals))
plt.hlines(y, 0, len(data[keys[0]])-1, colors='0.8', linestyles='dashed')

idx = np.argsort(y)
for i in reversed(idx):
	k = keys[i]
	plt.plot(data[k], label=k)

plt.legend()
plt.show()