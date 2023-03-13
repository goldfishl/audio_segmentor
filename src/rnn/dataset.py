import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio
import os

from src.utils import load_split # custom function to load data
from .config import train_config, device # custom configuration for training

class GenerationData(Dataset):
    """
    Dataset class for audio data used for speech segmentation.
    """
    def __init__(self, data_config):
        """
        Args:
            data_config (dict): Configuration for data loading.
        """
        self.data = load_split(data_config) # Load audio data and segmentation labels
        self.data_path = data_config['data_path'] # Path to audio data directory

    def __getitem__(self, n: int):
        """
        Get an audio file and its corresponding segmentation label at index `n`.

        Args:
            n (int): Index of audio file in dataset.

        Returns:
            tuple: A tuple of (fbanks, seg_label) where `fbanks` is a 2D tensor of Mel-frequency 
            cepstral coefficients (fbanks) and `seg_label` is a 1D tensor of segmentation labels.
        """
        audio_file, seg_label = self.data[n]
        audio_file = os.path.join(self.data_path, audio_file)
        audio, _ = torchaudio.load(audio_file) # Load audio file using torchaudio
        fbanks = train_config['feature'](audio) # Compute fbank using a specified feature extractor
        return fbanks, seg_label

    def __len__(self) -> int:
        """
        Get the number of audio files in the dataset.

        Returns:
            int: Number of audio files in the dataset.
        """
        return len(self.data)  


def collate_batch(batch):
    """
    Collate a batch of audio files into padded tensors.

    Args:
        batch (list): A list of tuples of (fbanks, seg_label) where `fbanks` is a 2D tensor of fbank 
        and `seg_label` is a 1D tensor of segmentation labels.

    Returns:
        tuple: A tuple of (fbanks, seq_len, seg_labels) where `fbanks` is a padded 3D tensor of fbank,
        `seq_len` is a tensor of sequence lengths, and `seg_labels` is a padded 2D tensor of segmentation
        labels.
    """
    fbanks, seq_len, seg_labels, = [], [], []
   
    for fbank,seg_label in batch:

        seq_len.append(fbank.shape[0]) # Save the sequence length of the fbank tensor
        fbanks.append(fbank) # Append fbank tensor to list
        seg_label = torch.tensor([int(char) for char in seg_label]) # Convert segmentation label to tensor of ints
        seg_labels.append(seg_label) # Append segmentation label tensor to list

    fbanks = pad_sequence(fbanks, batch_first=True) # Pad the fbank tensors to the same length
    seg_labels = pad_sequence(seg_labels, batch_first=True, padding_value = train_config['label_pad_value']) # Pad the segmentation label tensors to the same length
    
    seq_len = torch.tensor(seq_len) # Convert sequence lengths to a tensor

    return fbanks, seq_len, seg_labels