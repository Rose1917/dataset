'''
this is config file used to set the hypyer parameter like dataset name, batch size, learning rate, etc.
'''
import logging
import os
import torch
from string import Template
from utils import pos
from utils import prompt

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the dataset name
dataset_name = 'ae-100k'

# set the batch size
batch_size = 32

# set the learning rate
learning_rate = 2e-3

# set the number of epochs
num_epochs = 20

# set the base model
base_model = 'google-bert/bert-base-uncased'

# set the attribute to load
attributes = ['Material']

# prompt length
# prompt_length = 200

# hard_prompt_template
hard_prompt_template = Template('$candidate is a kind of [MASK].[SEP]')

# pos policy
pos_strategy = pos.PosStrategy.TOKEN

# optimizer
optimizer = torch.optim.AdamW

# shots
shots = [1, 5, 10, 20]

# none label
none_label = "none"

# prompt length
prompt_length = 100


# prompt init strategy
prompt_init_strategy = prompt.PromptInitStrategy.LABEL_INIT

# metrics
metrics = ["precision", "f1", "recall"]
