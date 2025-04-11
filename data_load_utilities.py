import pickle
import numpy as np
import torch


def load_data(data_folder):
    train_path = data_folder / 'posterior_train.pkl'
    train_inputs_path = data_folder / 'data_train.pkl'
    test_path = data_folder / 'posterior_test.pkl'
    test_inputs_path = data_folder / 'data_test.pkl'

    train_file = open(train_path, 'rb')
    train_data_info = pickle.load(train_file)
    train_file.close()

    train_inputs_file = open(train_inputs_path, 'rb')
    train_inputs_info = pickle.load(train_inputs_file)
    train_inputs_file.close()

    test_file = open(test_path, 'rb')
    test_data_info = pickle.load(test_file)
    test_file.close()

    test_inputs_file = open(test_inputs_path, 'rb')
    test_inputs_info = pickle.load(test_inputs_file)
    test_inputs_file.close()

    data_train = {
        'emissions': train_data_info['posterior'],
        'inputs': train_inputs_info['inputs'],
    }

    data_test = {
        'emissions': test_data_info['posterior'],
        'inputs': test_inputs_info['inputs'],
    }

    return data_train, data_test


def get_data_snips(data, num_time, stride=1):
    inputs_init_array = []
    inputs_array = []
    targets_array = []

    for emi, inp in zip(data['emissions'], data['inputs']):
        # convert the data to tensors
        emi_torch = torch.tensor(emi, dtype=torch.float32)
        inp_torch = torch.tensor(inp, dtype=torch.float32)

        # unfold it: this takes windows of size num_time from the data
        # use num_time + 2 so you can use the first time point for initial conditions
        # and the last time point as the target
        emi_snips = emi_torch.unfold(0, num_time + 2, stride)
        inp_snips = inp_torch.unfold(0, num_time + 2, stride)

        # the inputs to the RNN will be the concatenated measured data and the opto stimulation events
        inputs_array.append(torch.cat((emi_snips[:, :, 1:-1], inp_snips[:, :, 1:-1]), dim=1))
        inputs_init_array.append(emi_snips[:, :, 0:1])
        targets_array.append(emi_snips[:, :, 2:])

    data_snips = {
        'inputs': torch.cat(inputs_array, axis=0),
        'inputs_init': torch.cat(inputs_init_array, axis=0),
        'targets': torch.cat(targets_array, axis=0),
    }

    return data_snips

