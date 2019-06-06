from misc.constants import *

UNK = "*unknown*"
NULL = "*null*"

def _process_question(question):
	qstr = question.lower().strip()
	if qstr[-1] == "?":
		qstr = qstr[:-1]
	words = qstr.split()
	words = ["<s>"] + words + ["</s>"]
	return words

def _prepare_indices():
	import re
	import json
	from misc.util import Index
	from collections import defaultdict
	set_name = "train2014"
	MIN_COUNT = 10

	QUESTION_INDEX = Index()
	MODULE_INDEX = Index()
	MODULE_TYPE_INDEX = Index()
	ANSWER_INDEX = Index()

	UNK_ID = QUESTION_INDEX.index(UNK)
	MODULE_INDEX.index(UNK)
	ANSWER_INDEX.index(UNK)

	NULL_ID = QUESTION_INDEX.index(NULL)
	#MODULE_INDEX.index(NULL)
	#ANSWER_INDEX.index(NULL)

	word_counts = defaultdict(lambda: 0)
	with open(QUESTION_FILE % set_name) as question_f:
		questions = json.load(question_f)["questions"]
		for question in questions:
			words = _process_question(question["question"])
			for word in words:
				word_counts[word] += 1
	for word, count in word_counts.items():
		if count >= MIN_COUNT:
			QUESTION_INDEX.index(word)

	pred_counts = defaultdict(lambda: 0)
	with open(MULTI_PARSE_FILE % set_name) as parse_f:
		table = str.maketrans({'(':'', ')':'', ';':' '})
		for line in parse_f:
			parts = line.strip().translate(table).split()
			for part in parts:
				pred_counts[part] += 1
	for pred, count in pred_counts.items():
		if count >= 10 * MIN_COUNT:
			MODULE_INDEX.index(pred)

	answer_counts = defaultdict(lambda: 0)
	with open(ANN_FILE % set_name) as ann_f:
		annotations = json.load(ann_f)["annotations"]
		for ann in annotations:
			for answer in ann["answers"]:
				if answer["answer_confidence"] != "yes":
					continue
				word = answer["answer"]
				if re.search(r"[^\w\s]", word):
					continue
				answer_counts[word] += 1

	keep_answers = reversed(sorted([(c, a) for a, c in answer_counts.items()]))
	keep_answers = list(keep_answers)[:MAX_ANSWERS]
	for count, answer in keep_answers:
		ANSWER_INDEX.index(answer)

	return {
		'QUESTION' : QUESTION_INDEX,
		'MODULE' : MODULE_INDEX,
		'MODULE_TYPE' : MODULE_TYPE_INDEX,
		'ANSWER' : ANSWER_INDEX,
		'UNK_ID' : UNK_ID,
		'NULL_ID' : NULL_ID
	}

def _cached_load():
	import os
	import pickle
	INDEX_CACHE_FILE = 'cache/index.dat'
	if os.path.exists(INDEX_CACHE_FILE):
		with open(INDEX_CACHE_FILE, 'rb') as fd:
			index_data = pickle.load(fd)
	else:
		index_data = _prepare_indices()
		with open(INDEX_CACHE_FILE, 'wb') as fd:
			pickle.dump(index_data, fd, protocol=pickle.HIGHEST_PROTOCOL)
	return index_data

_idx = _cached_load()

QUESTION_INDEX = _idx['QUESTION']
MODULE_INDEX = _idx['MODULE']
MODULE_TYPE_INDEX = _idx['MODULE_TYPE']
ANSWER_INDEX = _idx['ANSWER']
UNK_ID = _idx['UNK_ID']
NULL_ID = _idx['NULL_ID']
