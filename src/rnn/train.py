import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn, optim
import time
import random
import os
import pandas as pd


from .dataset import GenerationData, collate_batch
from .config import train_config, valid_config, test_config, device
from .config import save_config, exp_config, setup_training_params
from src.utils import AverageMeter, calculate_stats
from .model import RNN
from tensorboardX import SummaryWriter


def train(model, train_loader, val_loader, writer):
    torch.set_grad_enabled(True)

    global_step, epoch = 1, 1
    save_config['metric']['Hparam/InSegment_Iou'] = 0
    best_epoch = 1
    best_val_stats = None

    # record the average training loss
    loss_meter = AverageMeter()    

    optimizer, scheduler, loss_fn = setup_training_params(model, writer)
    
    model.to(device)
    model.train()
    while epoch < exp_config['num_epochs'] + 1:
        for i, (fbanks, seq_len, seg_labels) in enumerate(train_loader):
            B = fbanks.shape[0]

            fbanks = fbanks.to(device)
            seg_labels = seg_labels.to(device)
            model_out = model(fbanks, seq_len)
        
            # reshapes the output tensor into a 2D tensor with the number of rows inferred from the tensor shape 
            # and the number of columns equal to the number of classes in the classification task.
            model_out = model_out.view(-1, model_out.shape[-1])
            seg_labels = seg_labels.view(-1)
            loss = loss_fn(model_out, seg_labels)

            # w
            if global_step <= exp_config['warmup_end'] and (global_step % exp_config['warmup_step'] == 0 or global_step == 1):
                optimizer.param_groups[0]['lr'] = global_step / exp_config['warmup_end'] * exp_config['lr']

            loss_meter.update(loss.item(), B)
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Loss/avg_train_step', loss_meter.avg, global_step)
            writer.add_scalar('Learning Rate/base_lr', optimizer.param_groups[0]['lr'], global_step)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config['clip_grad_norm'])
            optimizer.step()

            global_step += 1
        # validate
        stats = validate(model, val_loader, loss_fn, writer, epoch)

        if stats['class_wise_jaccard'][1] > save_config['metric']['Hparam/InSegment_Iou']:
            best_val_stats = stats
            best_epoch = epoch
            save_config['metric']['Hparam/InSegment_Iou'] = stats['class_wise_jaccard'][1]

            torch.save(model.state_dict(), save_config['best_model_path'])

        scheduler.step()

        writer.add_scalar('Loss/train_epoch', loss_meter.avg, epoch)

        loss_meter.reset()
        epoch += 1
            
        
    return best_val_stats, best_epoch
    
def validate( model, val_loader, loss_fn, writer, epoch=None, split='val'):

    loss_meter = AverageMeter()

    A_predictions = []
    A_targets = []
    
    with torch.no_grad():
    
        for i, (fbanks, seq_len, seg_labels) in enumerate(val_loader):
            B = fbanks.shape[0]

            fbanks = fbanks.to(device)
            seg_labels = seg_labels.to(device)
            model_out = model(fbanks, seq_len)
        
            # reshapes the output tensor into a 2D tensor with the number of rows inferred from the tensor shape 
            # and the number of columns equal to the number of classes in the classification task.
            _out = model_out.view(-1, model_out.shape[-1])
            _labels = seg_labels.view(-1)
            loss = loss_fn(_out, _labels)
            
            loss_meter.update(loss.item(), B)
            for (_out, _len, _label) in zip(model_out, seq_len, seg_labels):
                A_predictions.append(_out[:_len].to('cpu').detach().softmax(dim=1))
                A_targets.append(_label[:_len].to('cpu'))
        
        predictions = torch.cat(A_predictions, dim=0)
        targets = torch.cat(A_targets, dim=0)
        stats = calculate_stats(predictions, targets)
        stats['loss'] = loss_meter.avg

        writer.add_pr_curve(split.capitalize(), targets, predictions[:, 1], epoch)

        writer.add_scalar(f'Loss/{split}_epoch', loss_meter.avg, epoch)

        # Record the Segmentation Frame Accuracy
        writer.add_scalar(f'Accuracy/{split}_weighted', stats['weighted_acc'], epoch)
        writer.add_scalar(f'Accuracy/{split}_micro', stats['micro_acc'], epoch)
        writer.add_scalar(f'Accuracy/{split}_macro', stats['macro_acc'], epoch)
        writer.add_scalar(f'Accuracy/{split}_OutSegment', stats['class_wise_acc'][0], epoch)
        writer.add_scalar(f'Accuracy/{split}_InSegment', stats['class_wise_acc'][1], epoch)

        # Record the Segmentation Frame Recall
        writer.add_scalar(f'Recall/{split}_weighted', stats['weighted_recall'], epoch)
        writer.add_scalar(f'Recall/{split}_micro', stats['micro_recall'], epoch)
        writer.add_scalar(f'Recall/{split}_macro', stats['macro_recall'], epoch)
        writer.add_scalar(f'Recall/{split}_OutSegment', stats['class_wise_recall'][0], epoch)
        writer.add_scalar(f'Recall/{split}_InSegment', stats['class_wise_recall'][1], epoch)

        # Record the Segmentation Frame Precision
        writer.add_scalar(f'Precision/{split}_weighted', stats['weighted_precision'], epoch)
        writer.add_scalar(f'Precision/{split}_micro', stats['micro_precision'], epoch)
        writer.add_scalar(f'Precision/{split}_macro', stats['macro_precision'], epoch)
        writer.add_scalar(f'Precision/{split}_OutSegment', stats['class_wise_precision'][0], epoch)
        writer.add_scalar(f'Precision/{split}_InSegment', stats['class_wise_precision'][1], epoch)

        # Record the Segmentation Frame F1
        writer.add_scalar(f'F1/{split}_weighted', stats['weighted_f1'], epoch)
        writer.add_scalar(f'F1/{split}_micro', stats['micro_f1'], epoch)
        writer.add_scalar(f'F1/{split}_macro', stats['macro_f1'], epoch)
        writer.add_scalar(f'F1/{split}_OutSegment', stats['class_wise_f1'][0], epoch)
        writer.add_scalar(f'F1/{split}_InSegment', stats['class_wise_f1'][1], epoch)

        # Record the Segmentation Frame average precision
        writer.add_scalar(f'Average Precision/{split}_weighted', stats['weighted_ap'], epoch)
        writer.add_scalar(f'Average Precision/{split}_micro', stats['macro_ap'], epoch)

        # Record the IoU
        writer.add_scalar(f'IoU or Jaccard index/{split}_weighted', stats['weighted_jaccard'], epoch)
        writer.add_scalar(f'IoU or Jaccard index/{split}_micro', stats['micro_jaccard'], epoch)
        writer.add_scalar(f'IoU or Jaccard index/{split}_macro', stats['macro_jaccard'], epoch)
        writer.add_scalar(f'IoU or Jaccard index/{split}_OutSegment', stats['class_wise_jaccard'][0], epoch)
        writer.add_scalar(f'IoU or Jaccard index/{split}_InSegment', stats['class_wise_jaccard'][1], epoch)

        # Record the Dice
        writer.add_scalar(f'Dice/{split}_weighted', stats['weighted_dice'], epoch)
        writer.add_scalar(f'Dice/{split}_micro', stats['micro_dice'], epoch)
        writer.add_scalar(f'Dice/{split}_macro', stats['macro_dice'], epoch)
        writer.add_scalar(f'Dice/{split}_OutSegment', stats['class_wise_dice'][0], epoch)
        writer.add_scalar(f'Dice/{split}_InSegment', stats['class_wise_dice'][1], epoch)

        return stats


if __name__ == '__main__':
    os.makedirs(save_config['log_dir'], exist_ok=True)
    os.makedirs(save_config['hparam_log_dir'], exist_ok=True)

    writer = SummaryWriter(save_config['log_dir'], flush_secs=30)

    train_loader = DataLoader(GenerationData(train_config), batch_size=train_config['batch_size'], 
                          num_workers=train_config['num_workers'], pin_memory=False,
                          collate_fn=collate_batch, shuffle=True)
    val_loader = DataLoader(GenerationData(valid_config), batch_size=valid_config['batch_size'],
                        num_workers=valid_config['num_workers'], pin_memory=False,
                        collate_fn=collate_batch, shuffle=True)

    # initalizate model
    model = RNN()

    print('start training')
    print(save_config['best_model_path'])
    # train model
    best_val_stats, best_epoch = train(model, train_loader, val_loader, writer)

    test_loader = DataLoader(GenerationData(test_config), batch_size=test_config['batch_size'],
                        num_workers=test_config['num_workers'], pin_memory=False,
                        collate_fn=collate_batch, shuffle=True)
    
    sd = torch.load(save_config['best_model_path'])
    model.load_state_dict(sd, strict=False)

    test_stats = validate(model, test_loader, writer, epoch=best_epoch, split='test')
