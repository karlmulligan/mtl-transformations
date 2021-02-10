"""
%%% Input sentence property indicator functions %%%

Tell whether an input sentence has a certain grammatical / CONSTRUCTION
property; used to group by in analysis.

Should input a LIST of tokens (call with sent.split())
"""

# useful helper functions 
def is_singular(noun):
    return 1 if noun in ["newt", "orangutan", "peacock", "quail", "raven", "salamander", \
                            "tyrannosaurus", "unicorn", "vulture", "walrus", "xylophone", "yak", "zebra"] \
                            else 0
def is_plural(noun):
    return 1 if noun in ["newts", "orangutans", "peacocks", "quails", "ravens", "salamanders", \
                            "tyrannosauruses", "unicorns", "vultures", "walruses", "xylophones", "yaks", "zebras"] \
                            else 0

# useful lists (from grammarfile)
preps = ["around", "near", "with", "upon", "alongside", "behind", "above", "below"]
relativizers = ["who", "that"]
intrans_verbs = ["giggles", "smiles", "sleeps", "swims", "waits", "moves", "changes", "reads", "eats", \
            "giggle", "smile", "sleep", "swim", "wait", "move", "change", "read", "eat"] 
has_relativizer = lambda l: True if any(r in l for r in relativizers) else False
has_prep = lambda l: True if any(p in l for p in preps) else False 

#def subj_rc(sent):
#    if has_relativizer(sent):
#        if sent[2] in relativizers: 
#            return 1
#        else:
#            return 0
#    return 0
#
#def obj_rc(sent):
#    if has_relativizer(sent[3:]):
#        return 1
#    return 0
#
#def subj_pp(sent):
#    if has_prep(sent):
#        if sent[2] in preps:
#            return 1
#        else:
#            return 0
#    return 0
#
#def obj_pp(sent):
#    if has_prep(sent[3:]):
#        return 1
#    return 0
#
#def intransitive(sent):
#    if obj_rc(sent):
#        return 0
#    if sent[-3] in intrans_verbs:
#        return 1
#    return 0 
#     
#def token_passive(sent): 
#    if sent[-1] == "passive":
#        return 1
#    return 0
#
#def simple_patient(targ):
#    if "by" not in targ:
#        # intransitive sentence; make sure to exclude when looking at results
#        return 0
#    targ_half1 = targ[:targ.index("by")] 
#    if len(targ_half1) <= 4:
#        # only accepts sentences that start like e.g. "my zebra is accepted"
#        return 1
#    return 0
#
#def patient_distractor(targ):
#    """returns 1 only if the second NP in the patient NP clause is a different
#        plurality than the first NP"""
#    if "by" not in targ:
#        # intransitive sentence; make sure to exclude when looking at results
#        return 0
#    targ_half1 = targ[:targ.index("by")]
#    sing_count = 0
#    plural_count = 0
#    sing_count += sum([is_singular(word) for word in targ_half1])
#    plural_count += sum([is_plural(word) for word in targ_half1])
#
#    if sing_count == 1 and plural_count == 1:
#        return 1 
#    return 0
"""
%%% Evaluation metrics %%%

Each function operates on a per-sentence basis (all take input, target, and predicted 
sentences as their arguments, regardless of whether they use all of them or not.

These will be used to process the input/target/predicted sentences from running test.py.
That data will also have information about the input sentence (e.g. "subj rc", "obj pp",
"intransitive", etc.) which can then be used to group by for analysis.
"""

# all sentence evaluation metrics:

def full_right(inp, targ, pred):
    """(all sentences) how many sentences are fully correct?"""
    return 1 if targ.strip() == pred.strip() else 0

def full_except_period(inp, targ, pred):
    """(all sentences) how many sentences are fully correct except for a missing period?"""
    return 1 if targ.strip().split()[:-1] == pred.strip().split() else 0

def t_sent_len(inp, targ, pred):
    """length of target sentence"""
    return len(targ.split())

def p_sent_len(inp, targ, pred):
    """length of predicted sentence"""
    return len(pred.split())

def t_num_adj(inp, targ, pred):
    adjs = ["agreeable","bewildered","courageous","determined","exuberant","fantastic","grotesque","handsome"]
    num_adj = 0
    for w in targ.split():
        if w in adjs:
            num_adj += 1
    return num_adj

def p_num_adj(inp, targ, pred):
    adjs = ["agreeable","bewildered","courageous","determined","exuberant","fantastic","grotesque","handsome"]
    num_adj = 0
    for w in pred.split():
        if w in adjs:
            num_adj += 1
    return num_adj

def t_num_rc(inp, targ, pred):
    rels = ["who", "that"] 
    num_rels = 0
    for w in targ.split():
        if w in rels:
            num_rels += 1
    return num_rels

def p_num_rc(inp, targ, pred):
    rels = ["who", "that"] 
    num_rels = 0
    for w in pred.split():
        if w in rels:
            num_rels += 1
    return num_rels

def t_num_pp(inp, targ, pred):
    preps = ["around", "near", "with", "upon", "alongside", "behind", "above", "below"]
    num_preps = 0
    for w in targ.split():
        if w in preps:
            num_preps += 1
    return num_preps

def p_num_pp(inp, targ, pred):
    preps = ["around", "near", "with", "upon", "alongside", "behind", "above", "below"]
    num_preps = 0
    for w in pred.split():
        if w in preps:
            num_preps += 1
    return num_preps

def t_num_auxes(inp, targ, pred):
    return len([w for w in targ.split() if w in ["do", "does", "don't", "doesn't"]])
    
def p_num_auxes(inp, targ, pred):
    return len([w for w in pred.split() if w in ["do", "does", "don't", "doesn't"]])
    

# passivization-specific evaluation metrics:

def patient_right(inp, targ, pred):
    '''(passives) whether the noun phrase patient is correct'''
    targ_aux = "."
    pred_aux = "."
    for aux in ["is", "isn't", "are", "aren't"]:
        if aux in targ:
            targ_aux = aux
        if aux in pred:
            pred_aux = aux
    targ_patient = targ.split(targ_aux)[0]
    pred_patient = targ.split(pred_aux)[0]
    if targ_patient == pred_patient:
        return 1
    return 0 

def aux_right(inp, targ, pred):
    """(passives) whether auxes match"""
    for aux in ["is", "isn't", "are", "aren't"]:
        if aux in targ and aux in pred:
            return 1
    return 0


# question formation-specific evaluation metrics:

def first_aux_right(inp, targ, pred):
    return 1 if targ.split()[0] == pred.split()[0] else 0


# tense reinflection-specific evaluation metrics:

def matrix_aux_right(inp, targ, pred):
    targ_auxes = [w for w in targ.split() if w in ["do", "does", "don't", "doesn't", "did", "didn't"]]
    pred_auxes = [w for w in pred.split() if w in ["do", "does", "don't", "doesn't", "did", "didn't"]]
    #print("targ_auxes", targ_auxes)
    #print("pred_auxes", pred_auxes)
    if len(targ_auxes) != len(pred_auxes):
        return 0
    if len(targ_auxes) == 0 or len(pred_auxes) == 0:
        return 0
    if len(targ_auxes) == 1 and len(pred_auxes) == 1:
        if targ_auxes[0] == pred_auxes[0]:
            return 1
    t_matrix_aux = ""
    t_first_aux_pos = -1
    t_first_rel_pos = -1
    p_matrix_aux = ""
    p_first_aux_pos = -1
    p_first_rel_pos = -1
    for w, pos in zip(targ.split(), range(len(targ.split()))):
        if w in ["who", "that"] and t_first_rel_pos == -1:
            t_first_rel_pos = pos
        if w in ["do", "does", "don't", "doesn't", "did", "didn't"] and t_first_aux_pos == -1:
            t_first_aux_pos = pos
            t_matrix_aux = w
    for w, pos in zip(pred.split(), range(len(pred.split()))):
        if w in ["who", "that"] and p_first_rel_pos == -1:
            p_first_rel_pos = pos
        if w in ["do", "does", "don't", "doesn't", "did", "didn't"] and p_first_aux_pos == -1:
            p_first_aux_pos = pos
            p_matrix_aux = w
    #print(t_first_rel_pos, t_first_aux_pos, p_first_rel_pos, p_first_aux_pos) 
    if t_first_rel_pos == -1 or t_first_rel_pos > t_first_aux_pos: 
        return 1 if targ_auxes[0] == pred_auxes[0] else 0
    elif t_first_rel_pos < t_first_aux_pos and len(pred_auxes) > 1: 
        if p_first_rel_pos < p_first_aux_pos:
            return 1 if targ_auxes[1] == pred_auxes[1] else 0
    return 0
        
def num_right_auxes(inp, targ, pred):
    targ_auxes = [w for w in targ.split() if w in ["do", "does", "don't", "doesn't", "did", "didn't"]]
    pred_auxes = [w for w in pred.split() if w in ["do", "does", "don't", "doesn't", "did", "didn't"]]
    num = 0
    while targ_auxes and pred_auxes:
        if targ_auxes[0] == pred_auxes[0]:
            num += 1
        del targ_auxes[0]
        del pred_auxes[0]
    return num
   



#BELOW: no longer useful after refactor (just select by 'intransitive' when analyzing)

#def ungrammatical_right(inp, targ, pred):
#    """(passive INTRANSITIVE sentences) how many simple sentences are correctly marked ungrammatical?"""
#    return 1 if inp.split()[-1] == "passive" and targ.strip() == "[ungrammatical] ." and pred.strip() == "[ungrammatical] ." else 0
#
#def truncated_right(inp, targ, pred):
#    """(passives sentences NOT [ungrammatical]) how many sentences would be correct if passive 
#        were 'truncated' (i.e. had no by-phrase)"""
#    targ = targ.split()
#    pred = pred.split()
#
#    if "by" not in targ:
#        # intransitive sentence; make sure to exclude when looking at results
#        return 0
#
#    targ_trunc = targ[:targ.index("by")]
#    trunc_len = len(targ_trunc)
#
#    if len(pred) >= len(targ_trunc):
#        if targ_trunc == pred[:trunc_len]:
#            return 1
#    return 0
#
#def entire_np_order(inp, targ, pred):
#    """(passive sentences NOT [ungrammatical]) where order of both entire NPs are preserved"""
#    targ = targ.split()
#    pred = pred.split()
#    if "." in targ:
#        targ = targ[:-1]
#    if "." in pred:
#        pred = pred[:-1]
#    copula = "are" if "are" in targ else "is"
#    
#    if "by" not in targ:
#        # intransitive sentence; make sure to exclude when looking at results
#        return 0
#
#    targ_np1 = targ[:targ.index(copula)]
#    targ_np2 = targ[targ.index("by") + 1: -1]
#    
#    if "are" in pred:
#        pred_np1 = pred[:pred.index("are")]
#    elif "is" in pred:
#        pred_np1 = pred[:pred.index("is")]
#    else:
#        return 0 
#
#    if "by" in pred:
#        pred_np2 = pred[pred.index("by") + 1: -1]
#    else:
#        return 0
#
#    if targ_np1 == pred_np1 and targ_np2 == pred_np2:
#        return 1
#    return 0
#
#def main_dp_order(inp, targ, pred):
#    """(passive sentences NOT [ungrammatical]) where order of both main DP (e.g. "your salamander") 
#        are preserved -- even if the full NP is something like "your salamander that eats"""
#    targ = targ.split()
#    pred = pred.split()
#    
#    if "by" not in targ:
#        # intransitive sentence; make sure to exclude when looking at results
#        return 0
#
#    targ_half1 = targ[:targ.index("by")] 
#    targ_half2 = targ[targ.index("by")+1:] 
#
#    if "by" in pred:
#        pred_half1 = pred[:pred.index("by")]
#        pred_half2 = pred[pred.index("by")+1:]
#    else:
#        return 0
#
#    if targ_half1[:2] == pred_half1[:2] and targ_half2[:2] == pred_half2[:2]:
#        return 1
#    return 0
#
#def main_dp_patient_only(inp, targ, pred):
#    """(passive sentences NOT [ungrammatical]) where agent (i.e. from first NP)'s main DP (e.g. "your salamander") 
#        is correct, regardless of correctness of the rest of sentence"""
#    targ = targ.split()
#    pred = pred.split()
#
#    if len(targ) <= 2:
#        return 0
#
#    if targ[:2] == pred[:2]:
#        return 1
#    return 0
#
#
#def agreement(inp, targ, pred):
#    """(passive sentences NOT [ungrammatical]) where agreement with 'be' is correct"""
#
#    targ = targ.split()
#    pred = pred.split()
#
#    if "by" not in targ:
#        # intransitive sentence; make sure to exclude when looking at results
#        return 0
#
#    targ_half1 = targ[:targ.index("by")] 
#
#    if "by" in pred:
#        pred_half1 = pred[:pred.index("by")]
#    else:
#        return 0
#
#    np_patient = targ[1]
#    
#    if is_singular(np_patient):
#        if pred_half1[-2] == 'is':
#            return 1
#        else:
#            return 0
#    else:
#        if pred_half1[-2] == 'are':
#            return 1
#        else:
#            return 0
