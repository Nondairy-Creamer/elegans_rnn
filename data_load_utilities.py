import pickle
import numpy as np


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

