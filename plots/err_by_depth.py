import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

label_dict = dict(
	modular = 'Implementing neural data types',
	classic = 'Common end-to-end interfaces'
)
marker_dict = dict(
	modular = '^',
	classic = 'o'
)
color_dict = dict(
	modular = '#ff7f0e',
	classic = '#1f77b4'
)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('clevr_results')
	parser.add_argument('clevr_results_fulltrain')
	parser.add_argument('--on-validation', action='store_true')
	return parser.parse_args()

def main(args):

	min_depth = 3
	max_depth = 19
	if args.on_validation:
		max_depth = 18
	else:
		min_depth = 6

	with open(args.clevr_results) as fd:
		data = json.load(fd)
	with open(args.clevr_results_fulltrain) as fd:
		data_full = json.load(fd)

	fig, ax1 = plt.subplots()
	handles = []
	all_errors = list()
	for label in ['classic','modular']:

		# Results when trained up to length 5
		depths, errors = fetch_errors(data[label], min_depth, max_depth)
		all_errors.extend(errors)
		p = ax1.plot(depths, errors,
			label  = label_dict[label],
			marker = marker_dict[label],
			color  = color_dict[label]
		)

		# Results when trained on all lengths (baseline)
		depths, errors = fetch_errors(data_full[label], min_depth, max_depth)
		ax1.plot(depths, errors, 
			marker          = marker_dict[label],
			color           = color_dict[label],
			linestyle       = 'dotted',
			markerfacecolor = 'white'
		)

	if args.on_validation:
		# Separation mark between depths seen during training and deeper.
		ax1.plot([5.5,5.5], [min(all_errors), max(all_errors)], linestyle='dashed', color='black')
	plt.xticks(list(range(min_depth,max_depth+1)))

	ax1.set_xlabel('Program depth')
	ax1.set_ylabel('Mean error (%)')

	# Additional entry in legend
	handles, labels = ax1.get_legend_handles_labels()
	dashed_line = mlines.Line2D([],[], color='black', linestyle='dotted', label='Trained on all program depths')
	handles.append(dashed_line)

	ax1.legend(loc=None, handles=handles)
	plt.show()

def fetch_errors(values, min_depth, max_depth):
	depth, error_data = zip(*values.items())
	error, N = zip(*error_data)
	depth = [ int(d) for d in depth ]
	error = [ 100*e for e in error ]
	error = [ e for e,d in zip(error, depth) if d >= min_depth and d <= max_depth ]
	depth = [ d for d in depth if d >= min_depth and d <= max_depth ]
	return depth, error

if __name__ == '__main__':
	main(get_args())
