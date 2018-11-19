import os
import pandas as pd
import shutil
from random import randint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
def save_best_model(_is_best, _model, _model_save_path, _epoch):
    if _is_best:
        # TODO:save model
        save_path = os.path.join(_model_save_path, '%d_epoch_best_model.pth' % _epoch)
        torch.save(_model, save_path)
def onehot_encode(_label, _num_class):
    result = np.zeros(_num_class)
    result[_label] = 1
    return result
# other util
def accuracy(output, target):  # , topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    """
    correct = 0
    batch_size = target.size(0)
    for i in range(batch_size):
        tar = target[i].data.cpu().numpy()
        pred = output[i].data.cpu().numpy()
        if (tar) == np.argmax(pred):
            correct += 1
    return float(correct/batch_size)
def frame2_video_level_accuracy(_dic_video_level_preds, _dic_video_level_targets, _criterion):
    correct = 0
    num_videos = len(_dic_video_level_preds)
    video_level_preds = np.zeros((num_videos, 51))
    video_level_labels = np.zeros(num_videos)
    ii = 0
    for name in sorted(_dic_video_level_preds.keys()):
        preds = _dic_video_level_preds[name]
        label = _dic_video_level_targets[name]
        video_level_preds[ii, :] = preds
        video_level_labels[ii] = label
        ii += 1
        if np.argmax(preds) == (label):
            correct += 1
    # top1 top5
    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
    loss = _criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())
    return float(correct/num_videos), loss.data.cpu().numpy()
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
def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)
def record_info(info, filename, mode):
    if mode == 'train':
        result = (
            'Time {batch_time} '
            'Data {data_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '
            'Prec@5 {top5}\n'
            'LR {lr}\n'.format(batch_time=info['Batch Time'],
                               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],
                               lr=info['lr']))
        print(result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Data Time', 'Loss', 'Prec@1', 'Prec@5', 'lr']
    if mode == 'test':
        result = (
            'Time {batch_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '
            'Prec@5 {top5} \n'.format(batch_time=info['Batch Time'],
                                      loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))
        print(result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Loss', 'Prec@1', 'Prec@5']
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names)

def str2bool(bool_str):
    if bool_str.lower() == 'true':
        return True
    elif bool_str.lower() == 'false':
        return False
    else:
        raise ValueError

def make_save_dir(save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        pass
