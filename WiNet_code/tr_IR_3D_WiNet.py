import os
import glob
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
# from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger,CSVLogger
import data.trans as trans
import data.datasets as dataset

import numpy as np
from nn_util import *
from loss import *
import utils

from eval import  jacobian_determinant_vxm, process_label
from natsort import natsorted
os.environ["base_dir"] = '/bask/projects/d/duanj-ai-imaging/xxc/dataset_all'

def train_ixi():
    atlas_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/atlas.pkl')
    train_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/Train/')
    val_dir   = os.path.join(os.getenv('base_dir'), 'IXI_data/Val/')

    # tr_composed  = transforms.Compose([trans.RandomFlip(0),trans.NumpyType((np.float32, np.float32))])
    # trans.RandomFlip(0),
    # fast convergence
    tr_composed  = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    
    val_composed = transforms.Compose([trans.Seg_norm(),trans.NumpyType((np.float32, np.int16))])
    
    train_set = dataset.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=tr_composed)
    tr = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=32, pin_memory=True)
    
    val_set = dataset.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

    check_num = len(train_set)
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 1
    # args.precision = 32
    args.accelerator = 'gpu'
    args.device = 1
    args.default_root_dir = './logs'

    args.lr =  1e-4
    args.max_steps = 1000 * len(train_set)
     
    # args.log_every_n_steps = check_num
    args.use_dice_loss = False
    args.diff = True
    
    args.dice_val = utils.dice_IXI
    
    args.name = 'brain_ixi'
    v = 'WiNet-diff'

    args.val_check_interval=1/len(train_set) * check_num
    
    
    # ckpt_name = ''
    # args.resume_from_checkpoint = f'{args.default_root_dir}/{args.name}/{v}/checkpoints/{ckpt_name}.ckpt'

    logger = TensorBoardLogger(save_dir=args.default_root_dir, version=v, name=args.name, default_hp_metric=False)
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/dice_score', save_top_k=15, mode='max', 
                                     filename='val-e{epoch}-s{step}-dice{val/dice_score:.4f}-sim{val/loss_sim:.4f}', 
                                     auto_insert_metric_name=False, every_n_train_steps=check_num))

    from Model import UNet
    from Base_Model import BaseModel
    
    winet = UNet(2, 3, 8, wavelet='haar')
    WiNet = BaseModel(args=args, model=winet, som=2)
   
    WiNet.diff =  DiffeomorphicTransform(time_step=7)
    WiNet.loss_similarity = NCC_vxm()
    
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps, logger=logger,
                                            enable_progress_bar=False,log_every_n_steps=200)
    trainer.fit(WiNet, train_dataloaders=tr,  val_dataloaders=val)


if __name__ == "__main__":
    train_ixi()

