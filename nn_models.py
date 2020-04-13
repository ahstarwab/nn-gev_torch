import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class BLSTMMaskEstimator(nn.Module):
    def __init__(self, input_dim=513, hidden_dim=512, num_layers=1, dropout=0.3, bidirectional=True):
        super(BLSTMMaskEstimator, self).__init__()
        self.dropout = dropout
        # blstm_layer = SequenceBLSTM(513, 256, normalized=True)
        self.blstm_layer = nn.LSTM(input_dim, 256, num_layers, dropout=dropout, bidirectional=bidirectional)
        # relu_1 = SequenceLinear(256, 513, normalized=True)
        self.relu_1 = nn.Linear(hidden_dim, input_dim)
        # relu_2 = SequenceLinear(513, 513, normalized=True)
        self.relu_2 = nn.Linear(input_dim, input_dim)
        # noise_mask_estimate = SequenceLinear(513, 513, normalized=True)
        self.noise_mask_estimate = nn.Linear(input_dim, input_dim)
        # speech_mask_estimate = SequenceLinear(513, 513, normalized=True)
        self.speech_mask_estimate = nn.Linear(input_dim, input_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, Y):
        
        Y = Y.reshape(-1, 1, Y.shape[-1]) #[seq_len X 1 X input_dim]
        blstm, _ = self.blstm_layer(Y)
        
        relu_1 = self.relu_1(blstm)#, dropout=self.dropout)
        #TODO
        #Need torch.clamp(relu_1, min=0, max=1)?
        relu_2 = self.relu_2(relu_1)#, dropout=self.dropout)
        #TODO
        #Need torch.clamp(relu_2, min=0, max=1)
        X_mask = self.sigmoid(self.speech_mask_estimate(relu_2))
        N_mask = self.sigmoid(self.noise_mask_estimate(relu_2))
        

        return X_mask, N_mask

class SimpleFWMaskEstimator(nn.Module):
    def __init__(self, input_dim=513, hidden_dim=1024, output_dim = 513):
        super(SimpleFWMaskEstimator, self).__init__()
        self.relu_1 = nn.Linear(input_dim, hidden_dim)
        self.noise_mask_estimate = nn.Linear(hidden_dim, output_dim)
        self.speech_mask_estimate = nn.Linear(hidden_dim, output_dim)

    def forward(self, Y):
        relu_1 = self.relu_1(Y)
        #TODO
        #Need torch.clamp(relu_1, min=0, max=1)
        X_mask = nn.Sigmoid(self.speech_mask_estimate(relu_1))
        N_mask = nn.Sigmoid(self.noise_mask_estimate(relu_1))
        

        return X_mask, N_mask,