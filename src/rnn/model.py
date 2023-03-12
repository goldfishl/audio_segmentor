import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn


class RNN(nn.Module):
    def __init__(self, emb_dim=128, enc_hid_dim=256, out_dim=3, dropout=0.2):
        super().__init__()
        
        self.rnn = nn.GRU(input_size = emb_dim, 
                          hidden_size = enc_hid_dim, 
                          num_layers = 2,
                          dropout=dropout, 
                          batch_first=True)
        self.fc = nn.Sequential(
                      nn.Linear(enc_hid_dim,64),
                      nn.GELU(),
                      nn.Linear(64,32),
                      nn.GELU(),
                      nn.Linear(32,out_dim),
                  )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embed, seq_len=[], batch_infer=False):
        embed = self.dropout(embed)
        if batch_infer:
            packed_embed = pack_padded_sequence(embed, seq_len.to('cpu'), batch_first=True, enforce_sorted=False)
            packed_outputs, hidden = self.rnn(packed_embed)
            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embed)
        outputs = self.fc(outputs)
        return outputs