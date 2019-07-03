#!/usr/bin/env python2

import sexpdata

def parse_tree(p):
    if "'" in p:
        p = "none"
    parsed = sexpdata.loads(p)
    extracted = extract_parse(parsed)
    return extracted

def extract_parse(p):
    if isinstance(p, sexpdata.Symbol):
        return p.value()
    elif isinstance(p, int):
        return str(p)
    elif isinstance(p, bool):
        return str(p).lower()
    elif isinstance(p, float):
        return str(p).lower()
    return tuple(extract_parse(q) for q in p)

def parse_to_layout(parse):
    """ All leaves become find modules, all internal
    nodes become transform or combine modules dependent
    on their arity, and root nodes become describe
    or measure modules depending on the domain. """
    from misc.indices import FIND_INDEX, DESC_INDEX, UNK_ID
    from misc.util import ziplist
    if isinstance(parse, str):
        return "find", FIND_INDEX[parse] or UNK_ID
    head = parse[0]
    below = [ parse_to_layout(c) for c in parse[1:] ]
    modules_below, indices_below = ziplist(*below)
    module_head = 'and' if head == 'and' else 'describe'
    index_head = DESC_INDEX[head] or UNK_ID
    modules_here = [module_head] + modules_below
    indices_here = [index_head] + indices_below
    return modules_here, indices_here

def process_question(question):
    qstr = question.lower().strip()
    if qstr[-1] == "?":
        qstr = qstr[:-1]
    words = qstr.split()
    words = ["<s>"] + words + ["</s>"]
    return words
    
