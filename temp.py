from vqa import VQADataset
from misc.indices import ANSWER_INDEX, DESC_INDEX, QUESTION_INDEX
from collections import defaultdict

dataset = VQADataset('./', 'train2014')
yesno = { QUESTION_INDEX[s] for s in ['is', 'are', 'have', 'has', 'do', 'does'] }
choice = { QUESTION_INDEX[s] for s in ['or'] }
questions = defaultdict(lambda: set())

for i in range(len(dataset)):
	datum = dataset._by_id[dataset._id_list[i]]
	answers = set(datum['answers'])
	query = datum['layouts_indices'][0]

	q_words = set(datum['question'])
	q_particle = datum['question'][1]
	if q_particle in yesno and len(choice.intersection(q_words)) == 0:
		questions[q_particle].update(answers)

for q, answers in questions.items():
	answers = [ ANSWER_INDEX.get(a) for a in answers ]
	print(QUESTION_INDEX.get(q))
	print(answers)