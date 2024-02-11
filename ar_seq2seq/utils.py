import torch.nn as nn
import torch.nn.functional as F
import torch

# class ProjConcatLayer(nn.Module):
#     def __init__(self, d_model, d_hidden, dropout=0.0):
#         super().__init__()
#         self.input_to_hidden1 = nn.Linear(d_model, d_hidden)
#         self.input_to_hidden2 = nn.Linear(d_model, d_hidden)
#         # self.hidden_to_output = nn.Linear(d_hidden, d_output)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, input1, input2):
#         h = torch.cat([self.input_to_hidden1(input1), self.input_to_hidden2(input2)], dim=-1)
#         return h

class Dictionary(object):
    # helper class for extend the NAT base
    def __init__(self, num_codes):
        super().__init__()
        self.num_codes = num_codes

    def bos(self):
        return -1

    def eos(self):
        return -1

    def unk(self):
        return -1

    def pad(self):
        return -1

    def __len__(self):
        return self.num_codes


# class GateNet(nn.Module):
#     def __init__(self, d_model, d_hidden, d_output, dropout=0.0):
#         super().__init__()
#         self.input_to_hidden = nn.Linear(d_model, d_hidden)
#         self.hidden_to_output = nn.Linear(d_hidden, d_output)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, inputs):
#         h = F.relu(self.input_to_hidden(inputs))
#         h = self.dropout(h)
#         return self.hidden_to_output(h)
