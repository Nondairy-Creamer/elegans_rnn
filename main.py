import pickle
from pathlib import Path
import data_load_utilities as dlu
import torch
import torch.nn as nn

data_folder = Path('/home/mcreamer/Documents/data_sets/funcon_rnn/20240422_152517')
data_train, data_test = dlu.load_data(data_folder)

num_neurons = data_train['emissions'][0].shape[1]
input_size = num_neurons
batch_size = 5
num_layers = 1
num_time = 100

rnn = nn.LSTM(input_size, num_neurons, num_layers)
inputs = torch.randn(num_time, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, num_neurons)
c0 = torch.randn(num_layers, batch_size, num_neurons)
output, (hn, cn) = rnn(inputs, (h0, c0))

a=1