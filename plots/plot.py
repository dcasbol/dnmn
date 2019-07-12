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
plt.ylabel(meta['ylabel'])

# Most times, the xvals are the same for all plots shown
# "xlabel" : ["Time in seconds", "time"]
xkey = ''
xlabel = meta['xlabel']
if type(xlabel) == list:
	xlabel, xkey = xlabel
	xvals = data[xkey]

plt.xlabel(xlabel)

for key, values in data.items():
	if key == xkey: continue
	if xkey != '':
		assert type(values[0]) != list, "Cannot override global x values."
	elif type(values[0]) == list:
		xvals, values = zip(*values)
	else:
		xvals = [ 0.01 * i for i in range(len(values)) ]
	plt.plot(xvals, values, label=key)

plt.legend()
plt.show()