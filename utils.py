import os
import torch
import numpy as np

def to_th(x):
    return x if torch.is_tensor(x) else torch.from_numpy(x)

def to_np(x):
    val =  x.data.cpu().detach().numpy() if torch.is_tensor(x) else x
    return val  

def exists(x):
    return x is not None

def default_(val, d):
    if exists(val): return val
    return d() if callable(d) else d

def exists(val):
    return val is not None

def mod_(numer, denom):
    return (numer % denom) == 0

def tuple_(x, length = 1):
    return x if isinstance(x, tuple) else ((x,) * length)

def mkd(pth, tag=None):
    if tag: pth = os.path.join(pth, tag)
    
    os.makedirs(pth, exist_ok=True)
    return pth

def mkd(dir, fn, ext):
    os.makedirs(f'{dir}/', exist_ok=True)
    if os.path.exists(f'{dir}/{fn}.{ext}'):
        os.remove(f'{dir}/{fn}.{ext}')
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def dice_IXI(y_pred, y_true):
    # use same number of classes used in voxel_morph paper
    # ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebral-White-Matter', 'Cerebellum-White-Matter', 'Putamen',
    #  'VentralDC', 'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
    #  '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF', 'choroid-plexus']

    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2. * intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        idx += 1
    return np.mean(DSCs)