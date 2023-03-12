import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
from src.utils import load_split
from .config import train_config, device



class Acupuncture_Prescription(Dataset):
    def __init__(self, data_config):
        self._data = load_split(data_config)

    def __getitem__(self, n: int):
        return self._data[n]

    def __len__(self) -> int:
        return len(self._data)  


def collate_batch(batch):
    fbanks, seq_len, seg_labels, = [], [], []
   
    for _audio,_transcript in batch:
        _embed = train_config['feature'](_audio)
        seq_len.append(_embed.shape[0])
        fbanks.append(_embed)
        _transcript = torch.tensor([int(char) for char in _transcript])
        seg_labels.append(_transcript)

    fbanks = pad_sequence(fbanks, batch_first=True) # default padding_value=0
    seg_labels = pad_sequence(seg_labels, batch_first=True, padding_value = train_config['label_pad_value']) # 
    
    seq_len = torch.tensor(seq_len)

    return fbanks.to(device), seq_len.to("cpu"), seg_labels.to(device)
