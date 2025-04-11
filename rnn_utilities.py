import numpy as np
import torch


def predict(rnn, history, history_init, num_time):
    with torch.no_grad():
        device = history.device
        hist_length = history.shape[0]
        num_samples = history.shape[1]
        input_size = history.shape[2]
        predictions = torch.cat((history, torch.zeros(num_time, num_samples, input_size, device=device)), dim=0)

        # initialize the RNN and then run it forward
        h0 = torch.zeros_like(history_init)
        c0 = h0
        outputs = rnn(predictions[:hist_length, :, :], (h0, c0))[0]

        for t in range(1, num_time):
            h0 = torch.zeros_like(outputs[t:t+1, :, :])
            c0 = h0
            outputs = rnn(predictions[t: t + hist_length, :, :], (h0, c0))[0]
            predictions[hist_length + t, :, :outputs.shape[2]] = outputs[-1:, :, :]

        return predictions


