import torch
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_recall
from torchmetrics.functional.classification import multiclass_precision, multiclass_f1_score
from torchmetrics.functional.classification import multiclass_average_precision, multiclass_confusion_matrix
from torchmetrics.functional.classification import multiclass_jaccard_index
from torchmetrics.functional import dice
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def calculate_stats(predictions, targets):
    # Because there are many negative samples for each class, ROC and AUROC scores are not used.
    """Calculate statistics for classification task
    Args:
      output: a list of 2D tensor, [(samples_num, classes_num)]
      target: a list 1D tensor, [(samples_num)]
    Returns:
      stats: dict, statistics for classification task
    """

    
    stats = {}

    classes_num = predictions.shape[-1]

    # frame_accuracy
    stats['weighted_acc'] = multiclass_accuracy(predictions, targets, num_classes=classes_num, average='weighted')
    stats['macro_acc'] = multiclass_accuracy(predictions, targets, num_classes=classes_num, average='macro')
    stats['micro_acc'] = multiclass_accuracy(predictions, targets, num_classes=classes_num, average='micro')
    stats['class_wise_acc'] = multiclass_accuracy(predictions, targets, num_classes=classes_num, average=None)

    # recall
    stats['weighted_recall'] = multiclass_recall(predictions, targets, num_classes=classes_num, average='weighted')
    stats['macro_recall'] = multiclass_recall(predictions, targets, num_classes=classes_num, average='macro')
    stats['micro_recall'] = multiclass_recall(predictions, targets, num_classes=classes_num, average='micro')
    stats['class_wise_recall'] = multiclass_recall(predictions, targets, num_classes=classes_num, average=None)

    # precision
    stats['weighted_precision'] = multiclass_precision(predictions, targets, num_classes=classes_num, average='weighted')
    stats['macro_precision'] = multiclass_precision(predictions, targets, num_classes=classes_num, average='macro')
    stats['micro_precision'] = multiclass_precision(predictions, targets, num_classes=classes_num, average='micro')
    stats['class_wise_precision'] = multiclass_precision(predictions, targets, num_classes=classes_num, average=None)

    # f1 score
    stats['weighted_f1'] = multiclass_f1_score(predictions, targets, num_classes=classes_num, average='weighted')
    stats['macro_f1'] = multiclass_f1_score(predictions, targets, num_classes=classes_num, average='macro')
    stats['micro_f1'] = multiclass_f1_score(predictions, targets, num_classes=classes_num, average='micro')
    stats['class_wise_f1'] = multiclass_f1_score(predictions, targets, num_classes=classes_num, average=None)

    # average precision
    stats['weighted_ap'] = multiclass_average_precision(predictions, targets, num_classes=classes_num, average='weighted')
    stats['macro_ap'] = multiclass_average_precision(predictions, targets, num_classes=classes_num, average='macro')
    
    # IoU / Jaccard index
    # IoU(A, B) = |A ∩ B| / |A ∪ B| or IoU = TP / (TP + FP + FN - TP)
    stats['weighted_jaccard'] = multiclass_jaccard_index(predictions, targets, num_classes=classes_num, average='weighted')
    stats['macro_jaccard'] = multiclass_jaccard_index(predictions, targets, num_classes=classes_num, average='macro')
    stats['micro_jaccard'] = multiclass_jaccard_index(predictions, targets, num_classes=classes_num, average='micro')
    stats['class_wise_jaccard'] = multiclass_jaccard_index(predictions, targets, num_classes=classes_num, average=None)

    # dice coefficient
    # dice(A, B) = (2 * |A ∩ B|) / (|A| + |B|) or dice = (2 * TP) / (2 * TP + FP + FN)
    stats['weighted_dice'] = dice(predictions, targets, num_classes=classes_num, average='weighted')
    stats['macro_dice'] = dice(predictions, targets, num_classes=classes_num, average='macro')
    stats['micro_dice'] = dice(predictions, targets, num_classes=classes_num, average='micro')
    stats['class_wise_dice'] = dice(predictions, targets, num_classes=classes_num, average=None)

    # mAP@IoU to be implemented
    # Calculate the mean of the AP values at all IoU thresholds to obtain the mAP@IoU score.

    # mAP to be implemented
    # The mAP for multi-class object detection task is the mean of the AP values for each class over different IoU (Intersection over Union) thresholds. 
    # The most common IoU thresholds used are 0.5, 0.75, and 0.95, which are also referred to as IoU@[.5:.05:0.95].


    return stats