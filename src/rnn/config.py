from src.utils import wav2fbank, gener_config
from torch import nn, optim
import torch
import os
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
exp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
num_mel_bins = 128


exp_config = {
    'model_name': 'simple_rnn',
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 5e-7,
    'lrscheduler_start' : 5, 
    'lrscheduler_step' : 1,
    'lrscheduler_end' : 1000,
    'lrscheduler_gamma' : 0.9,  # normal scheduler every epoch
    'warmup_step' : 50,
    'warmup_end' : 1000,  # set -1 to disable warmup
    'num_epochs': 30,
    'loss_fn' : 'CrossEntropyLoss',
    'optimizer' : 'Adam',
    'clip_grad_norm' : 1.0,
}

# save config
save_config = {
    'log_dir' : os.path.join('logs', exp_config['model_name'], exp_name),
    'hparam_log_dir' : os.path.join('logs', 'hyperparam', exp_name), # comment for tensorboard
    'hparam_session_name' : exp_name,  # comment for tensorboard
    'best_model_path' : os.path.join('models', f'{ exp_name }_{ exp_config["model_name"] }_best.pth'),
    'worse_k' : 50,  # save the worse k recall class PR curve for analysis
    'metric' : {},  # record the result metrics for whole experiment in tensorboard
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


def setup_training_params(model, writer):
    """
    Set up the optimizer, loss function, and learning rate scheduler.

    Args:
        model: PyTorch neural network model.
        writer: SummaryWriter for logging training information.

    Returns:
        tuple: A tuple of (optimizer, loss_fn, scheduler) for training the model.
    """
    # Calculate statistics for the model
    save_config['metric']['Hparam/model_params'] = sum(p.numel() for p in model.parameters()) / 1e6
    trainables = [p for p in model.parameters() if p.requires_grad]
    writer.add_text('Model', 'Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    # Set optimizer
    if exp_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=exp_config['lr'], weight_decay=exp_config['weight_decay'], betas=(0.95, 0.999))

    # Set learning rate scheduler
    if exp_config['lrscheduler_start'] != -1:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, list(range(exp_config['lrscheduler_start'], exp_config['lrscheduler_end'], exp_config['lrscheduler_step'])), gamma=exp_config['lrscheduler_gamma'])

    # Set loss function
    if exp_config['loss_fn'] == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss(ignore_index=train_config['label_pad_value'])

    return optimizer, scheduler, loss_fn