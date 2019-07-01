from vqatorch import VQADataset
from misc.indices import ANSWER_INDEX, DESC_INDEX, QUESTION_INDEX

dataset = VQADataset('./', 'train2014')
yesno = { ANSWER_INDEX[s] for s in ['yes', 'no'] }
yesnomaybe = { ANSWER_INDEX[s] for s in ['yes', 'no', 'maybe', 'not sure', 'my best guess is yes', 'probably', 'possibly', 'no idea'] }
yesnoqueries = set()
multiuse = set()
questions = dict()

for i in range(len(dataset)):
	datum = dataset._by_id[dataset._id_list[i]]
	answers = set(datum['answers'])
	query = datum['layouts_indices'][0]

	if answers.intersection(yesnomaybe):
		try:
			questions[query].update(answers)
		except KeyError:
			questions[query] = answers

for q, answers in questions.items():
	answers = [ ANSWER_INDEX.get(a) for a in answers ]
	if len(yesnomaybe) + 5 > len(answers):
		print(DESC_INDEX.get(q))
		print(answers)