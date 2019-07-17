from misc.constants import *

UNK = "*unknown*"
NULL = "*null*"

class Index:
	def __init__(self):
		self.contents = dict()
		self.ordered_contents = []
		self.reverse_contents = dict()

	def __getitem__(self, item):
		if item not in self.contents:
			return None
		return self.contents[item]

	def index(self, item):
		if item not in self.contents:
			idx = len(self.contents) + 1
			self.ordered_contents.append(item)
			self.contents[item] = idx
			self.reverse_contents[idx] = item
		idx = self[item]
		assert idx != 0
		return idx

	def get(self, idx):
		if idx == 0:
			return "*invalid*"
		return self.reverse_contents[idx]

	def __len__(self):
		return len(self.contents) + 1

	def __iter__(self):
		return iter(self.ordered_contents)

def _process_question(question):
	qstr = question.lower().strip()
	if qstr[-1] == "?":
		qstr = qstr[:-1]
	words = qstr.split()
	words = ["<s>"] + words + ["</s>"]
	return words

def _index_words(index_obj, word_counts, min_count):
	selected = [ word for word, count in word_counts.items() if count >= min_count ]
	selected.sort() # Ensure same indexing in every system
	for word in selected:
		index_obj.index(word)

def _prepare_indices():
	import re
	import json
	from collections import defaultdict
	set_name = "train2014"
	MIN_COUNT = 10

	QUESTION_INDEX = Index()
	DESC_INDEX = Index()
	FIND_INDEX = Index()
	ANSWER_INDEX = Index()

	UNK_ID = QUESTION_INDEX.index(UNK)
	DESC_INDEX.index(UNK)
	FIND_INDEX.index(UNK)
	ANSWER_INDEX.index(UNK)

	NULL_ID = QUESTION_INDEX.index(NULL)

	word_counts = defaultdict(lambda: 0)
	with open(QUESTION_FILE % set_name) as question_f:
		for question in json.load(question_f)["questions"]:
			for word in _process_question(question["question"]):
				word_counts[word] += 1
	_index_words(QUESTION_INDEX, word_counts, MIN_COUNT)

	desc_counts = defaultdict(lambda: 0)
	find_counts = defaultdict(lambda: 0)
	with open(MULTI_PARSE_FILE % set_name) as parse_f:
		table = str.maketrans({'(':'', ')':'', ';':' '})
		for line in parse_f:
			parses = line.strip().split(';')
			for parse in parses:
				parts = line.strip().translate(table).split()
				desc_counts[parts[0]] += 1
				for part in parts[1:]:
					find_counts[part] += 1

	threshold = 10*MIN_COUNT
	_index_words(DESC_INDEX, desc_counts, threshold)
	_index_words(FIND_INDEX, find_counts, threshold)

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

	keep_answers = sorted([(c, a) for a, c in answer_counts.items()], reverse=True)
	keep_answers = list(keep_answers)[:MAX_ANSWERS]
	for count, answer in keep_answers:
		ANSWER_INDEX.index(answer)

	return {
		'QUESTION' : QUESTION_INDEX,
		'DESC' : DESC_INDEX,
		'FIND' : FIND_INDEX,
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
DESC_INDEX = _idx['DESC']
FIND_INDEX = _idx['FIND']
ANSWER_INDEX = _idx['ANSWER']
UNK_ID = _idx['UNK_ID']
NULL_ID = _idx['NULL_ID']

YESNO_QWORDS = { QUESTION_INDEX[w] for w in ['is', 'are', 'have', 'has', 'do', 'does'] }
OR_QWORD = QUESTION_INDEX['or']
NEG_ANSWERS = { ANSWER_INDEX[w] for w in ['no', '0', 'none', '*unknown*', 'no one', 'no idea'] }