import os
import glob
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import data.trans as trans
import data.datasets as dataset
import numpy as np
from nn_util import *
from loss import *
import utils
from eval import dice_IXI,jacobian_determinant_vxm
os.environ["base_dir"] = '/bask/projects/d/duanj-ai-imaging/xxc/dataset_all'

def infer_ixi(ckpt_name):
    atlas_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/atlas.pkl')
    test_dir =  os.path.join(os.getenv('base_dir'), 'IXI_data/Test/')

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 1
    args.accelerator = 'gpu'
    args.device = 1
    args.default_root_dir = './logs'
    
    args.compute_jac = False
    args.diff = True
    
    from Model import UNet
    from Base_Model import BaseModel
    winet = UNet(2, 3, 8, wavelet='haar', dwt_fn=2)
    WiNet = BaseModel(args=args, model=winet, som=2)
    
    loss_similarity = NCC_vxm()

    WiNet.load_state_dict(torch.load(ckpt_name)['state_dict'])
    WiNet = WiNet.cuda().eval()
    
    if args.diff:
        # WiNet.diff = DiffeomorphicTransform()
        WiNet.diff =  DiffeomorphicTransform2(time_step=7, shape=(160,192,224), device='cuda').eval() # fast slightly
     
    test_composed = transforms.Compose([trans.Seg_norm(),trans.NumpyType((np.float32, np.int16))])
    test_set = dataset.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    eval_dsc = utils.AverageMeter()
    eval_j   = utils.AverageMeter()
    
    with torch.no_grad():
        for data in test_loader:
            WiNet.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            flow =  WiNet(x.float(), y.float())
            flow = WiNet.diff(flow) if  args.diff else flow
            g, warped_lab = WiNet.stn(x_seg.float(), flow.movedim(1, -1), mod ='nearest')
            dice_score = dice_IXI(warped_lab, y_seg)
            eval_dsc.update(dice_score.item(), x.size(0))
            
            # _, warped = WiNet.stn(x, flow.movedim(1, -1))
            # loss_sim =  loss_similarity(y, warped)
            
            if args.compute_jac:
                dd, hh, ww = flow.shape[-3:]
                D_f_xy = flow.detach().cpu().numpy()
                D_f_xy[:,0,:,:,:] = D_f_xy[:,0,:,:,:] * dd / 2
                D_f_xy[:,1,:,:,:] = D_f_xy[:,1,:,:,:] * hh / 2
                D_f_xy[:,2,:,:,:] = D_f_xy[:,2,:,:,:] * ww / 2
            
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]
                jac_det = jacobian_determinant_vxm(D_f_xy[0, :, :, :, :])
                j = np.sum(jac_det <= 0)/np.prod(tar.shape)
                eval_j.update(j, x.size(0))
    
    print('DSC: {:.4f} +- {:.4f}'.format(eval_dsc.avg, eval_dsc.std))
    if args.compute_jac: print('J:   {:.4f} +- {:.4f}'.format(eval_j.avg, eval_j.std))
    

if __name__ == "__main__":
    ckpt_name = './ckpts/WiNet-diff-dice0.7522-sim-0.2389.ckpt'
    # ckpt_name = './ckpts/WiNet-dice0.7511-sim-0.2695.ckpt'
    infer_ixi(ckpt_name)
