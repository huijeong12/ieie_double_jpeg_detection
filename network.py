# -*- coding: utf-8 -*-

import math
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn 
from dctbasis import load_DCT_basis_torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=120):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TransformerWithCnnModel(nn.Module): 
    def __init__(self, input_size, conv_out_size, n_head, en_hidden_size, en_n_layers, drop_out):
        super(TransformerWithCnnModel, self).__init__()
        
        self.dct_basis = load_DCT_basis_torch().float()
        self.dct_basis = self.dct_basis.to(device)
 
        self.pos_encoder = PositionalEncoding(120, 0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=n_head, dim_feedforward=en_hidden_size, dropout=drop_out, activation='gelu', batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=en_n_layers)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=conv_out_size, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=conv_out_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_out_size, out_channels=conv_out_size, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=conv_out_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc1 = nn.Linear(12*120+64, 500)
        self.fc2 = nn.Linear(500+64, 2)
 
    def forward(self, x, qvectors, hidden=None):
        # feature extraction
        with torch.no_grad(): # 기록 추적 및 메모리 사용 방지
            gamma=1e+06 # 10^6
            x = F.conv2d(x, self.dct_basis, stride=8) 
            for b in range(-80, 81): 
                x_ = torch.sum(torch.sigmoid(gamma*(x-b)), axis=[2,3])/1024 
                x_ = torch.unsqueeze(x_, axis=1) 
                if b==-80:
                    features = x_
                else:
                    features = torch.cat([features, x_], axis=1)
            features = features[:, 0:160, :] - features[:, 1:161, :]
            features = torch.squeeze(features, axis=1)
            features = features[:, :, 1:64] # remove DC values
        
        output = features.transpose(1, 2)
        output = self.pos_encoder(output)
        output = self.encoder(output)

        output = self.conv1(features)
        output = self.conv2(output)

        output = torch.reshape(output, (-1, 12*120))

        output = torch.cat([qvectors, output], axis=1)
        output = F.relu(self.fc1(output))

        output = torch.cat([qvectors, output], axis=1)
        output = self.fc2(output)
    
        return output