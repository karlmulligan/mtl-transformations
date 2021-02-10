
# This code was adapted from the tutorial "Translation with a Sequence to 
# Sequence Network and Attention" by Sean Robertson. It can be found at the
# following URL:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# You must have PyTorch installed to run this code.
# You can get it from: http://pytorch.org/

# [Karl Mulligan]
# October 2019: This code was, in turn, adapted from Tom McCoy's github repo
# for similar experiments with subject-auxiliary inversion:
# https://github.com/tommccoy1/rnn-biases


# Imports
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sys
import os

# Functions for tracking time
import time
import math
from numpy import median
from numpy import mean

from datetime import date

from seq2seq_adj import pos_to_parse, sent_to_pos, file_to_batches

from evaluation import *
from models import *
from parsing import *

from cust_evals import *

from collections import OrderedDict
import csv

random.seed(7)

datestamp = str(date.today())

# Start-of-sentence and end-of-sentence tokens
# The standard seq2seq version only has one EOS. This version has 
# 2 EOS--one signalling that the original sentence should be returned,
# the other signalling it should be reversed.
# I use a 1-hot encoding for all tokens.
SOS_token = 0
EOS_tokenA = 1 # For ACTIVE
EOS_tokenB = 2 # For PASSIVE

prefix = "data/" + sys.argv[1] # This means we're using the language with agreement
directory = "models/" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + "_" + sys.argv[6]
data_dir = "data/"

counter = 0
dir_made = 0
# Reading the training data
trainingFile = prefix + '.train'
testFile = prefix + '.test'
devFile = prefix + '.dev'
genFile = prefix + '.gen'


batch_size = 5
if "TREE" in sys.argv[3]:
    batch_size = 1

MAX_LENGTH = 30

use_cuda = torch.cuda.is_available()

if use_cuda:
        available_device = torch.device('cuda')
else:
        available_device = torch.device('cpu')


auxes = ["can", "could", "will", "would", "do", "does", "don't", "doesn't"]

word2index = {}
index2word = {}


fi = open("index_ALL_adj.txt", "r")
for line in fi:
        parts = line.strip().split("\t")
        word2index[parts[0]] = int(parts[1])
        index2word[int(parts[1])] = parts[0]

#word2index["SOS"] = 0
#word2index["."] = 1
#word2index["?"] = 2
#index2word[0] = "SOS"
#index2word[1] = "."
#index2word[2] = "?"

MAX_LENGTH = 30


#train_batches, MAX_LENGTH = file_to_batches(trainingFile, MAX_LENGTH)
#dev_batches, MAX_LENGTH = file_to_batches(devFile, MAX_LENGTH)
test_batches, MAX_LENGTH = file_to_batches(testFile, MAX_LENGTH)
gen_batches, MAX_LENGTH = file_to_batches(genFile, MAX_LENGTH)

print(index2word)

recurrent_unit = sys.argv[3] # Could be "SRN" or "LSTM" instead
attention = sys.argv[4]# Could be "n" instead

if attention == "0":
        attention = 0
elif attention == "1":
        attention = 1
elif attention == "2":
	attention = 2
else:
        print("Please specify 'y' for attention or 'n' for no attention.")


MAX_EXAMPLE = 10000
MAX_LENGTH = 30

# Show the output for a few randomly selected sentences
def evaluateRandomly(encoder, decoder, batches, index2word, n=10):

    batch_size = batches[0][0].size()[1]

    for i in range(math.ceil(n * 1.0 / batch_size)):
        this_batch = random.choice(batches)
        
        input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
        target_sents = logits_to_sentence(this_batch[1], index2word)
        pred_sents = logits_to_sentence(evaluate(encoder, decoder, this_batch), index2word)

        for group in zip(input_sents, target_sents, pred_sents):
            print(group[0])
            print(group[1])
            print(group[2])
            print("")


# Where the actual running of the code happens
hidden_size = int(sys.argv[6]) # Default 128

print(MAX_LENGTH)
print("keys:", len(word2index.keys()))



if recurrent_unit == "TREE":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENC":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREEDEC":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEBOTH":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREENew":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENCNew":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREENOPRE":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENCNOPRE":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREEDECNOPRE":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEBOTHNOPRE":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREENewNOPRE":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENCNewNOPRE":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "ONLSTMPROC":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "ONLSTM", max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "ONLSTM", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
else:
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, recurrent_unit, max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), recurrent_unit, attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)


encoder1 = encoder1.to(device=available_device)
decoder1 = decoder1.to(device=available_device)


# For recording each observation for writing to dict later
obs_list = []
obs_cols = ["modelname", "task", "side_task", "side_task_strat", "modeltype", "attention", "learning_rate", "hidden_size", "random_seed", "datestamp", \
           "which_set", "input", "target", "pred", \
           "full_right", "full_except_period", "first_aux_right", \
           't_sent_len', 'p_sent_len', 't_num_adj' , 'p_num_adj' , 't_num_rc', 'p_num_rc', 't_num_pp', 'p_num_pp', 't_num_auxes', 'p_num_auxes']

counter = 0
direcs_to_process = 1

while direcs_to_process:
        if not os.path.exists(directory + "_" +  str(counter)):
                direcs_to_process = 0
        else:
                directory_now = directory + "_" + str(counter)
                counter += 1
		
                dec_list = sorted(os.listdir(directory_now))
                dec = sorted(dec_list[:int(len(dec_list)/2)], key=lambda x:float(".".join(x.split(".")[2:4])))[0]
                print("This directory:", dec)
                enc = dec.replace("decoder", "encoder")


                try:
                        encoder1.load_state_dict(torch.load(directory_now + "/" + enc))
                        decoder1.load_state_dict(torch.load(directory_now + "/" + dec))
                except RuntimeError:    
                        if len(word2index.keys()) == 81:
                                fi = open("index_tense.txt", "r")
                        else:
                                fi = open("index_ALL_adj.txt", "r")
                    
                        word2index = {}
                        index2word = {}


                        #fi = open("index.txt", "r")
                        for line in fi:
                                parts = line.strip().split("\t")
                                word2index[parts[0]] = int(parts[1])
                                index2word[int(parts[1])] = parts[0]

                        if recurrent_unit == "TREE":
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENC":
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREEDEC":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEBOTH":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREENew":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENCNew":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREENOPRE": 
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENCNOPRE":
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREEDECNOPRE":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEBOTHNOPRE":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREENewNOPRE":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENCNewNOPRE":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "ONLSTMPROC": 
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "ONLSTM", max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "ONLSTM", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        else: 
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, recurrent_unit, max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), recurrent_unit, attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)


                        encoder1 = encoder1.to(device=available_device)
                        decoder1 = decoder1.to(device=available_device)

                        encoder1.load_state_dict(torch.load(directory_now + "/" + enc))
                        decoder1.load_state_dict(torch.load(directory_now + "/" + dec))


                        
                print("Test")
                evaluateRandomly(encoder1, decoder1, test_batches, index2word)
                print("Gen")
                evaluateRandomly(encoder1, decoder1, gen_batches, index2word)
                print("Next")


                # EVALUATE ON TEST SET
                basic_ob = {"modelname" : directory.split("/")[1],
                            "task": sys.argv[2].split("_")[0],
                            "side_task": sys.argv[2].split("_")[1] if sys.argv[2].find("_") != -1 else "none",
                            "side_task_strat": "_".join(sys.argv[2].split("_")[1:]) if sys.argv[2].find("_") != -1 else "none",
                            "modeltype" : sys.argv[3],
                            "attention" : sys.argv[4],
                            "learning_rate" : sys.argv[5],
                            "hidden_size" : sys.argv[6],
                            "random_seed" : counter - 1,
                            "datestamp" : datestamp}

                for this_batch in test_batches:
                        input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
                        target_sents = logits_to_sentence(this_batch[1], index2word)
                        pred_sents = logits_to_sentence(evaluate(encoder1, decoder1, this_batch), index2word)

                        for trio in zip(input_sents, target_sents, pred_sents):
                            i, t, p = trio
                            this_ob = basic_ob.copy() 
                            this_ob['which_set'] = 'test'
                            # actual sentences
                            this_ob['input'] = i
                            this_ob['target'] = t
                            this_ob['pred'] = p
                            # evaluation metrics
                            this_ob['full_right'] = full_right(i, t, p)
                            this_ob['full_except_period'] = full_except_period(i, t, p)
                            this_ob['first_aux_right'] = first_aux_right(i, t, p) 
                            # sentence information
                            this_ob['t_sent_len'] = t_sent_len(i, t, p)
                            this_ob['p_sent_len'] = t_sent_len(i, t, p)
                            this_ob['t_num_adj'] = t_num_adj(i, t, p)
                            this_ob['p_num_adj'] = p_num_adj(i, t, p)
                            this_ob['t_num_rc'] = t_num_rc(i, t, p)
                            this_ob['p_num_rc'] = p_num_rc(i, t, p)
                            this_ob['t_num_pp'] = t_num_pp(i, t, p)
                            this_ob['p_num_pp'] = p_num_pp(i, t, p)
                            this_ob['t_num_auxes'] = t_num_auxes(i, t, p)
                            this_ob['p_num_auxes'] = p_num_auxes(i, t, p)

                            obs_list.append(this_ob)

                # EVALUATE ON GEN SET

                for this_batch in gen_batches:
                        input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
                        target_sents = logits_to_sentence(this_batch[1], index2word)
                        pred_sents = logits_to_sentence(evaluate(encoder1, decoder1, this_batch), index2word)

                        for trio in zip(input_sents, target_sents, pred_sents):
                            i, t, p = trio
                            this_ob = basic_ob.copy() 
                            this_ob['which_set'] = 'gen'
                            # actual sentences
                            this_ob['input'] = i
                            this_ob['target'] = t
                            this_ob['pred'] = p
                            # evaluation metrics
                            this_ob['full_right'] = full_right(i, t, p)
                            this_ob['full_except_period'] = full_except_period(i, t, p)
                            this_ob['first_aux_right'] = first_aux_right(i, t, p) 
                            # sentence information
                            this_ob['t_sent_len'] = t_sent_len(i, t, p)
                            this_ob['p_sent_len'] = t_sent_len(i, t, p)
                            this_ob['t_num_adj'] = t_num_adj(i, t, p)
                            this_ob['p_num_adj'] = p_num_adj(i, t, p)
                            this_ob['t_num_rc'] = t_num_rc(i, t, p)
                            this_ob['p_num_rc'] = p_num_rc(i, t, p)
                            this_ob['t_num_pp'] = t_num_pp(i, t, p)
                            this_ob['p_num_pp'] = p_num_pp(i, t, p)
                            this_ob['t_num_auxes'] = t_num_auxes(i, t, p)
                            this_ob['p_num_auxes'] = p_num_auxes(i, t, p)

                            obs_list.append(this_ob)                            

## kept for reference: which metrics should have which denominators
#testsets = ["test", "gen"]
#metrics = ["full_right", "full_except_period", "ungrammatical_right", "truncated_right", "entire_np_order", "main_dp_order", "main_dp_patient_only", "agr", "agr_simple", "agr_complex", "agr_has_distractor"]
#metric_which_total = ["total", "total", "ungrammatical_total", "passive_gram_total", "passive_gram_total", "passive_gram_total", "passive_gram_total", "passive_gram_total", "is_simple_patient_total", "is_complex_patient_total", "has_distractor_total"]

# write (append) results to csv
results_csv = "results/obs_results_question.csv".format(directory.split("/")[1])

with open(results_csv, "a+") as f:
    writer = csv.DictWriter(f, fieldnames=obs_cols)
    if os.stat(results_csv).st_size == 0:
        writer.writeheader()
    for ob in obs_list:
        writer.writerow(ob)
