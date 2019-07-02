import argparse
import torch
import json
from vqatorch import VQAEncoderDataset, encoder_collate_fn
from modules import QuestionEncoder
from misc.util import cudalize, majority_label
from misc.indices import ANSWER_INDEX, QUESTION_INDEX
from collections import defaultdict

if __name__ == '__main__':

	# ----------------------------------
	# PENDING INTEGRATION OF LSTM MODULE
	# Merge outputs with geom. average
	# ----------------------------------

	parser = argparse.ArgumentParser(description='Test the LSTM module alone')
	parser.add_argument('lstm_module', type=str)
	parser.add_argument('--output', type=str, default='results-lstm.json')
	args = parser.parse_args()

	dataset = VQAEncoderDataset()
	qenc = QuestionEncoder()

	qenc.load_state_dict(torch.load(args.lstm_module, map_location='cpu'))
	qenc = cudalize(qenc)

	answer_list = list()

	#for i in range(len(dataset)):
	for i in range(10):
		question, label = dataset[i]
		answers = dataset._by_id[dataset._id_list[i]]['answers']

		qtensor = cudalize(torch.tensor(question, dtype=torch.long)).unsqueeze(1)
		pred = qenc(qtensor)

		answer = ANSWER_INDEX.get(pred.argmax().item())

		readable = ' '.join([ QUESTION_INDEX.get(idx) for idx in question[1:-1] ])
		print(readable+'?', answer)
		print([ ANSWER_INDEX.get(a) for a in answers ])
		answer = majority_label(answers)
		print(ANSWER_INDEX.get(answer))

	quit()
	print('Writing results to {!r}'.format(args.output))
	with open(args.output, 'w') as fd:
		json.dump(answer_list, fd)
	print('Finished')

"""
Evaluate using code here in
https://github.com/GT-Vision-Lab/VQA
https://github.com/VT-vision-lab/VQA/blob/master/PythonEvaluationTools/vqaEvalDemo.py
"""