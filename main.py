from pathlib import Path
import data_load_utilities as dlu
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt


data_folder = Path('/home/mcreamer/Documents/data_sets/funcon_rnn/20240422_152517')
data_train, data_test = dlu.load_data(data_folder)

rng = np.random.default_rng(0)
num_time = 120
num_epochs = 10
stride = 5
learning_rate = 0.001
num_neurons = data_train['emissions'][0].shape[1]
batch_size = 1000
test_size = 5 * batch_size
input_size = 2 * num_neurons
rnn_dimension = num_neurons
num_layers = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_snips_train = dlu.get_data_snips(data_train, num_time, stride=stride)
data_snips_test = dlu.get_data_snips(data_test, num_time, stride=stride)

test_inputs = data_snips_test['inputs'].permute(2, 0, 1)
test_target = data_snips_test['targets'].permute(2, 0, 1)
test_samples = rng.choice(test_inputs.shape[1], test_size)
test_inputs = test_inputs[:, test_samples, :].to(device)
test_target = test_target[:, test_samples, :].to(device)

rnn = nn.LSTM(input_size, rnn_dimension, num_layers, bidirectional=False, device=device)
rnn.train()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

data_set = TensorDataset(data_snips_train['inputs'], data_snips_train['targets'])
loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
loss_array = []
test_loss_array = []
x_loc = []
counter = 0

for ep in range(num_epochs):
    for batch_data, batch_targets in loader:
        batch_data = batch_data.permute(2, 0, 1)
        batch_targets = batch_targets.permute(2, 0, 1)
        batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
        outputs, (hn, cn) = rnn(batch_data)
        loss = loss_fn(outputs[-1:, :, :], batch_targets)
        loss_array.append(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1

    outputs_test, (hn, cn) = rnn(test_inputs)
    loss_test = loss_fn(outputs_test[-1:, :, :], test_target)
    test_loss_array.append(loss_test.detach().cpu().numpy())
    optimizer.zero_grad()
    x_loc.append(counter)

    print(f'Epoch {ep+1} / {num_epochs}: Loss: {loss_array[-1]:.4f}')

loss_array = np.array(loss_array)
test_loss_array = np.array(test_loss_array)

plt.figure()
plt.plot(loss_array)
plt.plot(x_loc, test_loss_array)
plt.show()

a=1

