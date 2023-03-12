from src.utils import wav2fbank, gener_config
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_mel_bins = 128

exp_config = {
    'batch_size': 32,

}

train_config = {
    'batch_size': exp_config['batch_size'],
    'num_workers': 8,
    'feature': wav2fbank(num_mel_bins),
    'data_path': gener_config['data_path'],
    'split_file': gener_config['split_files']['train'],
    'label_pad_value' : 2,  # 0 is not in segment, 1 is in segment, 2 is padding
}

valid_config = train_config.copy()
valid_config['batch_size'] = 128
valid_config['split_file'] = gener_config['split_files']['valid']

test_config = valid_config.copy()
train_config['split_file'] = gener_config['split_files']['test']