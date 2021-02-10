# mtl-transformations
Code for "Structure Here, Bias There: Hierarchical Generalization by Jointly Learning Syntactic Transformations." The paper is available [here](https://arxiv.org). This code was largely adapated from the paper ["Does syntax need to grow on trees? Sources of hierarchical inductive bias in sequence to sequence networks"](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00304) ([repo](https://github.com/tommccoy1/rnn-hierarchical-biases)). 

Larger files, such as the datasets, models, and results files (with accompanying R notebook used for analysis and generating figures) can be downloaded separately at our OSF project site: https://osf.io/yrq3j/ 

## Requirements
This code was run using PyTorch version 0.4.1 with a GPU.

## Datasets

### Grammar 

The training and test data were created using a probabilistic context-free grammar (found in `generation/base.gr`). Each line has three values: a relative probabilty, a non-terminal, and its expansion. For instance, the following snippet indicates that a singular subject noun phrase `NP_M_S` expands to a simple DP, a DP modified by a prepositional phrase, or a DP modified by a relative clause, each with equal (1/3) probability.
```
1   NP_M_S  DP_S
1   NP_M_S  DP_S PP
1   NP_M_S  DP_S RC_S
```
Some expansion rules contain tags indicating subject and object boundaries, which are used to facilitate the dataset generation process.

### Datasets

The datasets for the first experiments (Section 4, dataset described in Section 3.2) ~~are already available in `data/`~~ are avilable at [our OSF project website](https://osf.io/yrq3j/), due to size constraints. The naming convention is as follows: `targettask_sidetask_sidetaskrule.set`. For example, target task question formation with side task passivization using hierarchical rule MOVE-PATIENT uses `question_passive_patient.train` as a training set. For the full list of transformations and their hierarchical and linear rules, refer to Section 3.1. There are four sets per task combination: a training set, a development set, an in-distribution test set, and an out-of-distribution generalization set.

#### Dataset generation
To generate the datasets yourself, you can run `generation/make_sets.py`, which takes 3 arguments: the name of the `.raw` file containing many samples of input sentences generated from the PCFG, one of three transformations (`passive`, `question`, `tense`), and whether the sentences are all ambiguous between linear and hierarchical rules (`amb`), containing unambiguous evidence for a hierarchical rule (`unamb`), or containing unambiguous evidence for a linear rule (`unamb_lin`). 
```
python make_sets.py [raw file] [transformation] [rule]
```
For example `python make_sets.py deep passive unamb` will produce four files: `passive_patient.train`, `passive_patient.dev`, `passive_patient.test`, and `passive_patient.gen` (the latter two not used for our experiments, sicne we test only on ambiguous data). The size of the these sets can be modified at the beginning of the script.

#### Multitask learning datasets
To create the datasets used for the multitask learning models, we simply concatenate and shuffle the training and development sets for two transformations -- one ambiguous (e.g. `question.train`) and one unambiguous (e.g. `passive_patient.train`) -- to get a multitask dataset (`question_passive_patient.train`), a task which is accomplished by running `generation/catshuf.sh` (or `generation/catshuf_multitrans.sh`, for multitask datasets with one ambiguous transformation and two unambiguous ones). 

#### Few-shot learning datasets
To create the datasets used for the few-shot learning experiments, in which a few disambiguating examples are introduced into the training sets, you can run `data/make_nshot_sets.sh` to create datasets with 5, 10, 50, 100, 500, and 1000 examples taken from the (disambiguating example only) generalization set and inserted into the training set. These sets have the same names as the original sets, with `_n-shot` appended at the end: e.g. `question_passive_patient_100-shot.train` for a version with 100 disambiguating examples in the training set.


## Training
To train a model, run `seq2seq.py`. For instance, to train a multitask model with question formation as a target task and passivization with MOVE-PATIENT as a side task, run the following code:

```
python3 seq2seq.py question_passive_patient question_passive_patient GRU 2 0.001 256
```

For more information on the parameters in this command, please refer to Tom McCoy's original Github for [more detailed instructions](https://github.com/tommccoy1/rnn-hierarchical-biases#basic-description-of-the-code).

To replicate the experiments in the paper, use the above hyperparameters and run that command for each task combination 10 times. Models will be saved in `models/` (the trained models used in the paper are available in this repo's `models/`. 

## Evaluation
To evaluate a model, run `test_[task].py` (or `test_[task]_nshot.py`, for few-shot models), replacing `[task]` with the name of the target task transformation: `passive`, `question`, or `tense`. For example, to test the model we trained in the line above, run the following:

```
python3 test_question.py question_passive_patient question_passive_patient GRU 2 0.001 256

```

This will produce a tidy data `.csv` file containing every input, target (hierarchical) output, and model-predicted ouptut for that model on the test and generalization sets, along with model metadata. This can be used in R or elsewhere to get summary statistics or to perform further analyses. Since these files can get very large (~1.5GB), please go to [our OSF project website](https://osf.io/yrq3j/) to download them, along with the R notebook used to perform the analyses and generate the figures used in the paper.

## Questions

For any questions or comments about this code, please contact [karl.mulligan@jhu.edu](mailto:karl.mulligan@jhu.edu). 
