from pathlib import Path
import data_load_utilities as dlu
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt
import rnn_utilities as ru


# load in the data from our trained model folder
data_folder = Path('/home/mcreamer/Documents/data_sets/funcon_rnn/20240422_152517')
data_train, data_test = dlu.load_data(data_folder)

# parameters, eventually should put these in a configuration file
rng_seed = 1
rng = np.random.default_rng(rng_seed)
num_time = 120
num_epochs = 5
stride = 5
learning_rate = 0.01
num_neurons = data_train['emissions'][0].shape[1]
batch_size = 2**10
test_size = 5 * batch_size
input_size = 2 * num_neurons
rnn_dimension = num_neurons
num_layers = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# cut the data up into segments of length num_time for training and evaluation
# shape is (samples, neurons + opto inputs, num_time)
data_snips_train = dlu.get_data_snips(data_train, num_time, stride=stride)
data_snips_test = dlu.get_data_snips(data_test, num_time, stride=1)

# grab some samples from the test data set for evaluation
# LSTM requires inputs as (time, samples, inputs) so swap here
test_inputs = data_snips_test['inputs'].permute(2, 0, 1)
test_inputs_init = data_snips_test['inputs_init'].permute(2, 0, 1)
test_targets = data_snips_test['targets'].permute(2, 0, 1)
test_samples = rng.choice(test_inputs.shape[1], test_size)
test_inputs_sampled = test_inputs[:, test_samples, :].to(device)
test_inputs_init_sampled = test_inputs_init[:, test_samples, :].to(device)
test_targets_sampled = test_targets[:, test_samples, :].to(device)

# create the LSTM and optimizer
rnn = nn.LSTM(input_size, rnn_dimension, num_layers=num_layers, bidirectional=False, device=device)
# rnn = nn.SimpleRnn(input_size, rnn_dimension, num_layers=num_layers, bidirectional=False, device=device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# use the data loader from pytorch to make batching easier
data_set = TensorDataset(data_snips_train['inputs'], data_snips_train['inputs_init'], data_snips_train['targets'])
loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

# save the losses in a list
loss_array = []
test_loss_array = []
x_loc = []  # this is to plot the test lost alongside the train loss
counter = 0

for ep in range(num_epochs):
    h0 = torch.zeros_like(test_inputs_init_sampled)
    c0 = h0
    outputs_test, (hn, cn) = rnn(test_inputs_sampled, (h0, c0))
    loss_test = loss_fn(outputs_test, test_targets_sampled)
    test_loss_array.append(loss_test.detach().cpu().numpy())
    x_loc.append(counter)

    for batch_data, batch_data_init, batch_targets in loader:
        # (samples, inputs, num_time) -> (num_time, samples, inputs)
        batch_data = batch_data.permute(2, 0, 1).to(device)
        batch_data_init = batch_data_init.permute(2, 0, 1).to(device)
        batch_targets = batch_targets.permute(2, 0, 1).to(device)

        h0 = torch.zeros_like(batch_data_init)
        c0 = h0
        outputs = rnn(batch_data, (h0, c0))[0]
        loss = loss_fn(outputs, batch_targets)
        loss_array.append(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1

    print(f'Epoch {ep+1} / {num_epochs}: Loss: {loss_array[-1]:.4f}')

loss_array = np.array(loss_array)
test_loss_array = np.array(test_loss_array)

plt.figure()
plt.plot(loss_array)
plt.plot(x_loc, test_loss_array)
plt.show()

# get the occasion of each of the opto events
num_predict = 60
stim_loc = torch.any(test_inputs[-1, :, rnn_dimension:], dim=1)

pred_inputs = test_inputs[:, stim_loc, :].to(device)
pred_inputs_init = test_inputs_init[:, stim_loc, :].to(device)
pred_targets = test_targets[:, stim_loc, :].to(device)

predictions = ru.predict(rnn, pred_inputs, pred_inputs_init, num_predict).to('cpu').numpy()

plt.figure()
plt.plot(predictions[:, :10, 0])
plt.show()

a=1

