"""Genuine spatial ConvLSTM cell and bidirectional sequence wrapper."""
import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels,
                               kernel_size=kernel_size, padding=padding)

    def forward(self, x, state):
        hidden, cell = state
        ingate, forgetgate, candidate, outgate = self.gates(torch.cat((x, hidden), dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(forgetgate) * cell + torch.sigmoid(ingate) * torch.tanh(candidate)
        hidden = torch.sigmoid(outgate) * torch.tanh(cell)
        return hidden, cell

    def init_state(self, batch, height, width, reference):
        shape = (batch, self.hidden_channels, height, width)
        return reference.new_zeros(shape), reference.new_zeros(shape)


class BidirectionalConvLSTM(nn.Module):
    """Accepts B,T,C,H,W and returns B,T,2*hidden,H,W."""
    def __init__(self, input_channels, hidden_channels=64):
        super().__init__()
        self.forward_cell = ConvLSTMCell(input_channels, hidden_channels)
        self.backward_cell = ConvLSTMCell(input_channels, hidden_channels)

    def forward(self, sequence):
        if sequence.ndim != 5:
            raise ValueError("ConvLSTM input must have shape (batch, time, channels, height, width)")
        batch, steps, _, height, width = sequence.shape
        state = self.forward_cell.init_state(batch, height, width, sequence)
        forward = []
        for index in range(steps):
            state = self.forward_cell(sequence[:, index], state)
            forward.append(state[0])
        state = self.backward_cell.init_state(batch, height, width, sequence)
        backward = [None] * steps
        for index in range(steps - 1, -1, -1):
            state = self.backward_cell(sequence[:, index], state)
            backward[index] = state[0]
        return torch.cat((torch.stack(forward, dim=1), torch.stack(backward, dim=1)), dim=2)
