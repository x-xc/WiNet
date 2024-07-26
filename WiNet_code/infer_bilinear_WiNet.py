import os
import glob
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import transforms

import data.trans as trans
import data.datasets as dataset

import numpy as np
from nn_util import *
from loss import *
import utils
import eval
from natsort import natsorted
os.environ["base_dir"] = '/bask/projects/d/duanj-ai-imaging/xxc/dataset_all'

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


def infer_b_ixi(ckpt_name):
    atlas_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/atlas.pkl')
    test_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/Test/')
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = dataset.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    
    
    model_folder = f'./ckpts/'
    log_dir = './Quantitative_Results/'
    tag = 'WiNet-diff'
    
    csv_name = ckpt_name + '_Val'  if 'Val' in test_dir else ckpt_name

    dict =  eval.process_label()
    
    mkd(log_dir, csv_name, 'csv')
    csv_writter(tag, log_dir + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', log_dir + csv_name)
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 1
    # args.precision = 32
    args.accelerator = 'gpu'
    args.device = 1
    args.default_root_dir = './logs'
    args.lr =  1e-4
    args.diff = True

    from Model import UNet
    from Base_Model import BaseModel
    
    winet = UNet(2, 3, 8, wavelet='haar', dwt_fn=2)
    WiNet = BaseModel(args=args, model=winet, som=2)

    model_path = os.path.join(model_folder, ckpt_name)
    print(model_path)
  
    best_model = torch.load(model_path)['state_dict']
    WiNet.load_state_dict(best_model)
    WiNet = WiNet.cuda().eval()
    
    if args.diff:
        # WiNet.diff =  DiffeomorphicTransform()
        WiNet.diff =  DiffeomorphicTransform2(time_step=7, shape=(160,192,224), device='cuda').eval() # fast slightly
    

    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            WiNet.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            flow =  WiNet(x.float(), y.float())
            flow = WiNet.diff(flow) if  args.diff else flow
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            
            x_segs = []
            for i in range(46):
                def_seg = WiNet.stn(x_seg_oh[:, i:i + 1, ...].float(), flow.movedim(1,-1).float(),  mod = 'bilinear')[1]
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            
            dd, hh, ww = flow.shape[-3:]
            D_f_xy = flow.detach().cpu().numpy()
            D_f_xy[:,0,:,:,:] = D_f_xy[:,0,:,:,:] * dd / 2
            D_f_xy[:,1,:,:,:] = D_f_xy[:,1,:,:,:] * hh / 2
            D_f_xy[:,2,:,:,:] = D_f_xy[:,2,:,:,:] * ww / 2
            
            jac_det =  eval.jacobian_determinant_vxm(D_f_xy[0, :, :, :, :])
            line =  eval.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line + ','+ str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, log_dir + csv_name)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans =  eval.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw =  eval.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))


if __name__ == "__main__":
    # ckpt_name = 'WiNet-dice0.7511-sim-0.2695.ckpt'
    ckpt_name = 'WiNet-diff-dice0.7522-sim-0.2389.ckpt'
    
    infer_b_ixi(ckpt_name)
