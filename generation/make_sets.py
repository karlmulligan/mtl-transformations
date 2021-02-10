import random
import sys

train_size = 100000
test_size = 10000
dev_size = 1000
gen_size = 10000

prefix = sys.argv[1]    # Should be a .raw (tagged) file
trans = sys.argv[2]     # passive, question, tense
amb = sys.argv[3]       # amb, unamb, unamb_lin
#amb = True if amb == "amb" else False
unamb_dict = {  "passive" : "_patient",
                "question" : "_main",
                "tense" : "_subject"}
unamb_lin_dict = {  "passive" : "_second",
                    "question" : "_first",
                    "tense" : "_recent"}
amb_suffix = ""
if amb == "unamb":
    amb_suffix = unamb_dict[trans]
if amb == "unamb_lin":
    amb_suffix = unamb_lin_dict[trans]

# Tags wrap around subject and object NPs.
# Possible tags: [Subj], [Obj_S], [Obj_P]
remove_tags = True

fi = open(prefix + ".raw", "r")


fo_train = open(trans + amb_suffix +  ".train", "w")
fo_dev = open(trans + amb_suffix +  ".dev", "w")
fo_test = open(trans + amb_suffix + ".test", "w")
fo_gen = open(trans + amb_suffix + ".gen", "w")

used_dict = {}

count_train = 0
count_test = 0
count_dev = 0
count_gen = 0

# Depends on grammar file, of course.
# Participle forms are those that come after the auxilliary in a passive, i.e. sweep --> "was SWEPT by"
# Only transitive verbs need have participles for our purposes.
participle_dict_s = {"entertains" : "entertained",
                   "amuses" : "amused",
                   "high_fives" : "high_fived",
                   "applauds" : "applauded",
                   "confuses" : "confused",
                   "admires" : "admired",
                   "accepts" : "accepted",
                   "remembers" : "remembered",
                   "comforts" : "comforted"}

participle_dict_p = {"entertain" : "entertained", 
                   "amuse" : "amused",
                   "high_five" : "high_fived",
                   "applaud" : "applauded",
                   "confuse" : "confused",
                   "admire" : "admired",
                   "accept" : "accepted",
                   "remember" : "remembered",
                   "comfort" : "comforted"
                  }

participle_dict = {**participle_dict_s, **participle_dict_p}

s_inflect_dict = {y:x for x,y in participle_dict_s.items()}

p_inflect_dict = {y:x for x,y in participle_dict_p.items()}

verb_list = list(participle_dict.keys())

aux_list = ["does", "doesn't", "do", "don't"]

aux_inflect_dict = {"does" : "did",
                    "do" : "did",
                    "doesn't" : "didn't",
                    "don't" : "didn't"}

rels = ["who", "that"]
preps = ["around", "near", "with", "upon", "alongside", "behind", "above", "below"]
s_noun_list = ["newt", "orangutan", "peacock", "quail", "raven", "salamander", "tyrannosaurus", "unicorn", "vulture", "walrus", "xylophone", "yak", "zebra"]
p_noun_list = ["newts", "orangutans", "peacocks", "quails", "ravens", "salamanders", "tyrannosauruses", "unicorns", "vultures", "walruses", "xylophones", "yaks", "zebras"]


# Helper functions

def remove_tags(sent_list):
    new_sent = []
    is_tag = lambda x: True if x[0] == "[" and x[-1] == "]" else False
    for token in sent_list:
        if not is_tag(token):
            new_sent.append(token)
    return new_sent


def extract_tag(sent_list, tagname):
    b_idx = -1 if "["+tagname+"]" not in sent_list else sent_list.index("["+tagname+"]")
    if b_idx == -1:
        return None
    e_idx = sent_list[b_idx+1:].index("["+tagname+"]") + b_idx
    return sent_list[b_idx+1:e_idx+1]


def get_disamb_flag(sent_list, trans):
    """ Takes a sent with tags as a list and returns True if:
            for passives:   the second noun is embedded in a subject PP or subject RC, 
                            *AND* is different from the patient noun phrase.
            for quest:      there is an RC in the subject, *AND* the auxes are different
            for tense:      there is a second noun in a subject PP or subject RC,
                            *AND* is different plurality than main noun phrase
    """
    if trans == "passive":
        subj_toks = extract_tag(sent_list, "Subj")
        # second np
        emb_np = []
        if "who" in subj_toks or "that" in subj_toks:
            rel_idx = [idx for idx, word in enumerate(subj_toks) if word in ["who", "that"]][0]
            if subj_toks[rel_idx + 1] in aux_list:
                # "verb (aux) first"
                emb_np = subj_toks[rel_idx + 3:]
            else:
                # "verb last"
                emb_np = subj_toks[rel_idx + 1:-2]
        emb_prep_list = [word for word in subj_toks if word in preps]
        if len(emb_prep_list) > 0:
            prep_idx = [idx for idx, word in enumerate(subj_toks) if word in preps][0]
            emb_np = subj_toks[prep_idx + 1:]

        patient_np = []
        if "[Obj_S]" in sent_list:
            patient_np = extract_tag(sent_list, "Obj_S")
        else:
            patient_np = extract_tag(sent_list, "Obj_P")

        if emb_np != [] and emb_np != patient_np:
            #print("in get_disamb_flag:", emb_np, patient_np)
            return True
        return False

    elif trans == "question":
        subj_list = extract_tag(sent_list, "Subj")
        emb_aux = None
        for w in subj_list:
            if w in aux_list:
                emb_aux = w
        main_aux_index = sent_list.index("[Subj]",1) + 1
        if emb_aux and emb_aux != sent_list[main_aux_index]:
            return True
        return False

    elif trans == "tense":
        subj_list = extract_tag(sent_list, "Subj")
        noun1 = None
        noun1_plurality = 0 
        noun2 = None
        noun1_plurality = 0
        for w in subj_list:
            if not noun1:
                if w in s_noun_list:
                    noun1 = w
                    noun1_plurality = 1
                elif w in p_noun_list:
                    noun1 = w
                    noun1_plurality = 2
            else:
                if w in s_noun_list:
                    noun2 = w
                    noun2_plurality = 1
                if w in p_noun_list:
                    noun2 = w
                    noun2_plurality = 2
        if noun2 and noun1_plurality != noun2_plurality:
            return True
        return False

    else:
        print("in get_disamb_flag: unknown transformation")
    return False



# Transformation functions

def activize(sent):
    active_sent = []

    active_sent = remove_tags(sent.split())

    active_sent = " ".join(active_sent)

    return active_sent + " ACTIVE\t" + active_sent + "\n", get_disamb_flag(sent.split(), "passive")


def passivize(sent, participle_dict):
    passive_sent = []
    
    sent_list = sent.split()

    subj_toks = []
    obj_toks = []
    sing_obj = True

    subj_toks = extract_tag(sent_list, "Subj")
    obj_toks = extract_tag(sent_list, "Obj_S")
    if obj_toks is None:
        obj_toks = extract_tag(sent_list, "Obj_P")
        sing_obj = False
    if obj_toks is None:
        #i.e. sentence has an intransitive verb: ungrammatical
        this_sent_type = "ung"
        if "who" in sent_list or "that" in sent_list:
            this_sent_type = "ung_rc"
        return " ".join(remove_tags(sent_list)) + " PASSIVE\t" + "[UNGRAMMATICAL] . \n", this_sent_type
        #print("Error: trying to passivize a sentence without appropriate obj tags, e.g. [Obj_S]")

    verb_idx = (sent_list.index("[Obj_S]") - 1) if sing_obj else (sent_list.index("[Obj_P]") - 1)
    verb = sent_list[verb_idx]
    
    aux_idx = (sent_list.index("[Obj_S]") - 2) if sing_obj else (sent_list.index("[Obj_P]") - 2)
    aux = sent_list[aux_idx]

    copula = "is" if sing_obj else "are"
    if aux in ["don't", "doesn't"]:
        copula = "isn't" if sing_obj else "aren't"

    participle = participle_dict[verb]

    passive_sent = obj_toks + [copula] + [participle] 
    passive_sent.append(".")

    passive_sent = " ".join(passive_sent)

    return " ".join(remove_tags(sent_list)) + " PASSIVE\t" + passive_sent + "\n", get_disamb_flag(sent.split(), "passive")


def make_pass_lin(sent, participle_dict):
    passive_sent = []
    
    sent_list = sent.split()

    subj_toks = []
    obj_toks = []
    sing_obj = True

    subj_toks = extract_tag(sent_list, "Subj")
    obj_toks = extract_tag(sent_list, "Obj_S")
    if obj_toks is None:
        obj_toks = extract_tag(sent_list, "Obj_P")
        sing_obj = False
    if obj_toks is None:
        #i.e. sentence has an intransitive verb: ungrammatical
        this_sent_type = "ung"
        if "who" in sent_list or "that" in sent_list:
            this_sent_type = "ung_rc"
        return " ".join(remove_tags(sent_list)) + " PASSIVE\t" + "[UNGRAMMATICAL] . \n", this_sent_type

    verb_idx = (sent_list.index("[Obj_S]") - 1) if sing_obj else (sent_list.index("[Obj_P]") - 1)
    verb = sent_list[verb_idx]
    
    aux_idx = (sent_list.index("[Obj_S]") - 2) if sing_obj else (sent_list.index("[Obj_P]") - 2)
    aux = sent_list[aux_idx]

    copula = "is" if sing_obj else "are"
    if aux in ["don't", "doesn't"]:
        copula = "isn't" if sing_obj else "aren't"

    participle = participle_dict[verb]

    passive_sent = obj_toks + [copula] + [participle] 
    passive_sent.append(".")

    passive_sent = " ".join(passive_sent)

    # OVERRIDE to pick second NP if the subject has a PP or RC
    singulars = ["newt", "orangutan", "peacock", "quail", "raven", "salamander", "tyrannosaurus", "unicorn", "vulture", "walrus", "xylophone", "yak", "zebra"]
    plurals = ["newts", "orangutans", "peacocks", "quails", "ravens", "salamanders", "tyrannosauruses", "unicorns", "vultures", "walruses", "xylophones", "yaks", "zebras"]
    emb_np = []
    verb = ""
    emb_verb = "" 
    emb_aux = ""
    copula_positivity = True
    if get_disamb_flag(sent_list, "passive"):
        if "who" in subj_toks or "that" in subj_toks:
            rel_idx = [idx for idx, word in enumerate(subj_toks) if word in ["who", "that"]][0]
            if subj_toks[rel_idx + 1] in aux_list:
                # "verb (aux) first"
                emb_np = subj_toks[rel_idx + 3:]
                emb_verb = subj_toks[rel_idx + 2]
            else:
                # "verb last"
                emb_np = subj_toks[rel_idx + 1:-2]
                emb_verb = subj_toks[-1]
        else:
            # i.e. embedded PP
            prep_idx = [idx for idx, word in enumerate(subj_toks) if word in preps][0]
            emb_np = subj_toks[prep_idx + 1:]
        # for RCs, use embedded verb; for PPs use matrix verb
        # for RCs, use embedded aux for polarity; for PPs use matrix aux for polarity (i.e. first)
        if emb_verb != "":
            verb = emb_verb
            emb_aux = [word for word in subj_toks if word in aux_list][0]
            if emb_aux in ["don't", "doesn't"]:
                copula_positivity = False
        else:
            verb = sent_list[sent_list.index("[Subj]",1) + 2]
            matrix_aux = sent_list[sent_list.index("[Subj]",1) + 1]
            if matrix_aux in ["don't", "doesn't"]:
                copula_positivity = False
        participle = participle_dict[verb]
        emb_noun = [word for word in emb_np if word in singulars + plurals][0]
        if emb_noun in singulars:
            if copula_positivity:
                copula = "is"
            else:
                copula = "isn't"
        else:
            if copula_positivity:
                copula = "are"
            else:
                copula = "aren't"
        passive_sent = emb_np + [copula] + [participle] 
        passive_sent.append(".")

        passive_sent = " ".join(passive_sent)
    return " ".join(remove_tags(sent_list)) + " PASSIVE\t" + passive_sent + "\n", get_disamb_flag(sent.split(), "passive")
      


def make_decl(sent):
    decl_sent = " ".join(remove_tags(sent.split()))

    return decl_sent + " DECL\t" + decl_sent + "\n", get_disamb_flag(sent.split(), "question")

def make_quest(sent):
    q_sent = []
    sent_list = sent.split()
    aux_index = sent_list.index("[Subj]",1) + 1
    verb_index = aux_index + 1
    obj_list = []
    if "[Obj_S]" in sent_list:
        obj_list = extract_tag(sent_list, "Obj_S")
    else:
        obj_list = extract_tag(sent_list, "Obj_P")

    q_sent.append(sent_list[aux_index])
    q_sent.extend(extract_tag(sent_list, "Subj"))
    q_sent.append(sent_list[verb_index])
    q_sent.extend(obj_list)
    q_sent.append("?")
    q_sent = " ".join(q_sent)

    return " ".join(remove_tags(sent_list)) + " QUEST\t" + q_sent + "\n", get_disamb_flag(sent.split(), "question")

def make_quest_lin(sent):
    #a.k.a. move-first
    q_sent = []
    sent_list = remove_tags(sent.split())
    these_auxes = [word for word in sent_list if word in aux_list]
    first_aux = these_auxes[0]
    idx = sent_list.index(these_auxes[0])
    q_sent = sent_list.copy()
    q_sent = q_sent[:-1]
    del q_sent[idx]
    q_sent = [first_aux] + q_sent
    q_sent.append("?")
    q_sent = " ".join(q_sent)

    return " ".join(sent_list) + " QUEST\t" + q_sent + "\n", get_disamb_flag(sent.split(), "question")
    


### past --> past (identity) and past --> present
### need to make past for both functions

def make_past(sent):
    sent_list = sent.split()
    past_sent = sent_list
    aux_index = sent_list.index("[Subj]",1) + 1
   
    for w_idx in range(len(past_sent)):
        if past_sent[w_idx] in aux_inflect_dict.keys():
            past_sent[w_idx] = aux_inflect_dict[sent_list[w_idx]]
    past_sent = remove_tags(past_sent)

    return " ".join(past_sent) + " PAST\t" + " ".join(past_sent) + "\n", get_disamb_flag(sent.split(), "tense")


def make_present(sent):
    present_sent = " ".join(remove_tags(sent.split()))
    sent_list = sent.split()
    past_sent = sent_list
    aux_index = sent_list.index("[Subj]",1) + 1
    
    #past_sent[aux_index] = aux_inflect_dict[sent_list[aux_index]]
    # NOTE: assumes transforming all auxes in sentence
    for w_idx in range(len(past_sent)):
        if past_sent[w_idx] in aux_inflect_dict.keys():
            past_sent[w_idx] = aux_inflect_dict[sent_list[w_idx]]
    past_sent = remove_tags(past_sent)

    return " ".join(past_sent) + " PRESENT\t" + present_sent + "\n", get_disamb_flag(sent.split(), "tense")


def make_present_lin(sent):
    # ak.k.a. inflect-recent: inflect based on number of most recent (last in subj tags) noun
    sent_list = sent.split()

    prev_i = 0
    prev_prev_i = 0
    for i, word in enumerate(sent_list):
        if word in aux_list:
            noun_list = [w for w in sent_list[prev_i+1:i] if w in s_noun_list + p_noun_list]
            if len(noun_list) == 0:
                noun_list = [w for w in sent_list[prev_prev_i+1:i] if w in s_noun_list + p_noun_list]
            if noun_list[-1] in s_noun_list and word == "do":
                sent_list[i] = "does"
            elif noun_list[-1] in s_noun_list and word == "don't":
                sent_list[i] = "doesn't"
            elif noun_list[-1] in p_noun_list and word == "does":
                sent_list[i] = "do"
            elif noun_list[-1] in p_noun_list and word == "doesn't":
                sent_list[i] = "don't"
            prev_prev_i = prev_i
            prev_i = i
    present_sent = " ".join(remove_tags(sent_list))

    past_sent = sent_list
    matrix_aux_index = sent_list.index("[Subj]",1) + 1

    for w_idx in range(len(past_sent)):
        if past_sent[w_idx] in aux_inflect_dict.keys():
            past_sent[w_idx] = aux_inflect_dict[sent_list[w_idx]]
    past_sent = remove_tags(past_sent)

    return " ".join(past_sent) + " PRESENT\t" + present_sent + "\n", get_disamb_flag(sent.split(), "tense")
    


# MAIN
choose_trans = False

for line in fi:
    if count_train >= train_size and count_test >= test_size:
        break

    raw_sent = line.strip()
    if raw_sent in used_dict:
        continue

    used_dict[raw_sent] = 1

    sent = ""
    disamb_flag = False

    choose_trans = not choose_trans

    if trans == "passive":
        if choose_trans:
            if amb == "unamb_lin":
                sent, disamb_flag = make_pass_lin(raw_sent, participle_dict)
            else:
                sent, disamb_flag = passivize(raw_sent, participle_dict)
        else:
            sent, disamb_flag = activize(raw_sent)
    elif trans == "question":
        if choose_trans:
            if amb == "unamb_lin":
                sent, disamb_flag = make_quest_lin(raw_sent)
            else:
                sent, disamb_flag = make_quest(raw_sent)
        else:
            sent, disamb_flag = make_decl(raw_sent) 
    elif trans == "tense":
        if choose_trans:
            if amb == "unamb_lin":
                sent, disamb_flag = make_present_lin(raw_sent)
            else:
                sent, disamb_flag = make_present(raw_sent)
        else:
            sent, disamb_flag = make_past(raw_sent) 
    else:
        print("error")
    
    if count_gen < gen_size:
        if disamb_flag and choose_trans:
            fo_gen.write(sent)
            count_gen += 1
    elif count_test < test_size:
        if amb == "amb" and disamb_flag and choose_trans:
            # then we can't have it in the regular test set!
            choose_trans = False
            continue
        else:
            #print("Writing to fo_test:", disamb_flag, "\t", sent)
            fo_test.write(sent)
            count_test += 1
    elif count_dev < dev_size:
        if amb == "amb" and disamb_flag and choose_trans:
            # then we can't have it in the dev set!
            choose_trans = False
            continue
        else:
            #print("Writing to fo_dev:", disamb_flag, "\t", sent)
            fo_dev.write(sent)
            count_dev += 1
    elif count_train < train_size:
        if amb == "amb" and disamb_flag and choose_trans:
            choose_trans = False
            continue
        else:
            fo_train.write(sent)
            count_train += 1
    else:
        break 
#       elif disamb_flag == "obj_rc" and choose_trans:
#           fo_gen.write(sent)
#           count_gen += 1
#           count_orc += 1
#    elif choose == 0 and count_test < test_size and (not rel_on_subj or not quest):
#        if not rel_on_subj or not quest:
#            fo_test.write(result)
#            count_test += 1
#    elif count_train < train_size:
#        fo_train.write(sent)
#        count_train += 1
#    else:
#        break

print(count_gen, test_size)
