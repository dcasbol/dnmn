import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='Display benchmark data')
parser.add_argument('jsonpath', type=str)
args = parser.parse_args()

with open(args.jsonpath) as fd:
	meta, data = json.load(fd)

plt.figure()

plt.title(meta['title'])
plt.xlabel(meta['xlabel'])
plt.ylabel(meta['ylabel'])

for key, values in data.items():
	plt.plot(values, label=key)

plt.legend()
plt.show()