import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils import *
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Norm, Act
from monai.networks.blocks import UpSample
from monai.utils import InterpolateMode, UpsampleMode
from monai.networks.layers.utils import get_act_layer

Act.add_factory_callable("softsign", lambda: nn.modules.Softsign)

        
def conv(in_c, out_c, k=3, s=1, p=1, dim=3, bias=True, 
         act='prelu', norm=None, drop=None):
    return Convolution(
            spatial_dims=dim,
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            strides=s,
            padding=p,
            adn_ordering="NA" if exists(norm) else "A",
            act=act,
            dropout=drop,
            norm=norm,
            bias=bias,
        )

def up_sample(in_c, out_c, dim=3, train=True, act='prelu', s=2, align=True, mode='nearest'):
    if not train: act = None
    if mode =='nearest': align = None
#   mode = ['nearest', InterpolateMode.BILINEAR, InterpolateMode.TRILINEAR][dim-1]
    return nn.Sequential(UpSample(spatial_dims=dim, in_channels=in_c, out_channels=out_c,
                                scale_factor=s, kernel_size=2, size=None,
                                mode=UpsampleMode.DECONV if train else UpsampleMode.NONTRAINABLE,
                                pre_conv='default', 
                                interp_mode=mode, align_corners=align,
                                bias=True,
                                apply_pad_pool=True),
                                get_act_layer(act) if exists(act) else nn.Identity())


class conv_twice(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, s=1, act='prelu', norm=None):
        super(conv_twice, self).__init__()
        self.conv1 = conv( in_ch, out_ch, 3, s, 1, dim=dim, act=act, norm=norm)
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1, dim=dim, act=act, norm=norm)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x    

class LayerNorm(torch.nn.Module):
    def __init__(self, dim):
        
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        d,h,w = x.shape[-3:]
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm(x)
        return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)    
    

       
class Up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, 
                skip_c=0, act='prelu', norm=None, 
                train=True, mode='nearest', align=True):
        super(Up_conv, self).__init__()
        self.skip_c = skip_c
        self.up = up_sample(in_ch, in_ch, dim=dim, train=train, act=act, mode=mode, align=align)
        self.conv1 = conv(in_ch + skip_c, out_ch, 3, 1, 1, dim=dim, act=act, norm=norm)
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1, dim=dim, act=act, norm=norm)
    
    def forward(self, x, skip_in=None):
        x = torch.cat((self.up(x), skip_in), 1) if self.skip_c>0 else self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    
class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)], indexing='ij')
        grid_h = grid_h.to(flow.device).float()
        grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:,:,:,:,0]
        flow_h = flow[:,:,:,:,1]
        flow_w = flow[:,:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = F.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return sample_grid, warped
    
class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)], indexing='ij')
        grid_h = grid_h.to(flow.device).float()
        grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)
        
        for i in range(self.time_step):
            flow_d = flow[:,0,:,:,:]
            flow_h = flow[:,1,:,:,:]
            flow_w = flow[:,2,:,:,:]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
    
         
            flow = flow + F.grid_sample(flow, deformation, mode='bilinear', padding_mode="border", align_corners = True)
        
        return flow
         
class DiffeomorphicTransform2(nn.Module):
    def __init__(self, shape, time_step=7, device = 'cuda:0'):
        super(DiffeomorphicTransform2, self).__init__()
        self.time_step = time_step
        d2, h2, w2 = shape
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)], indexing='ij')
        grid_h = grid_h.to(device).float()
        grid_d = grid_d.to(device).float()
        grid_w = grid_w.to(device).float()
        self.grid_d = nn.Parameter(grid_d, requires_grad=False)
        self.grid_w = nn.Parameter(grid_w, requires_grad=False)
        self.grid_h = nn.Parameter(grid_h, requires_grad=False)
    
    def forward(self, flow):
        flow = flow / (2 ** self.time_step)
        
        for i in range(self.time_step):
            flow_d = flow[:,0,:,:,:]
            flow_h = flow[:,1,:,:,:]
            flow_w = flow[:,2,:,:,:]
            disp_d = (self.grid_d + flow_d).squeeze(1)
            disp_h = (self.grid_h + flow_h).squeeze(1)
            disp_w = (self.grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border", align_corners = True)
        
        return flow
    
    

class STN(nn.Module):
    def __init__(self,  device='cuda', norm=True):
        super(STN, self).__init__()
        self.dev = device
        self.norm = norm
    
    def reference_grid(self, shape):
        dhw = shape[2:]
        if self.norm: 
            grid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, s) for s in dhw], indexing='ij'), dim=0)
        else: 
            grid = torch.stack(torch.meshgrid([torch.arange(0, s) for s in dhw], indexing='ij'), dim=0)
        grid = nn.Parameter(grid.float(), requires_grad=False).to(self.dev)
        return grid
    
    def forward(self, image, ddf, mode='bilinear', p_mode='zeros'):
        spatial_dims = len(image.shape) - 2
        grid =  ddf + self.reference_grid(image.shape)
        if not self.norm:
            for i in range(len(spatial_dims)):
                grid[:, i, ...] = 2 * (grid[:, i, ...] / (spatial_dims[i] - 1) - 0.5)
        grid = grid.movedim(1, -1)
        idx_order = list(range(spatial_dims - 1, -1, -1))
        grid = grid[..., idx_order]  # z, y, x -> x, y, z
        warped = F.grid_sample(image, grid, mode=mode, padding_mode=p_mode, align_corners=True)
        return grid, warped
    
class svf_exp(nn.Module):
    def __init__(self, time_step=7, device ='cuda'):
        super(svf_exp,self).__init__()
        self.time_step = time_step
        self.warp = STN(device=device)
    def forward(self, flow):
        flow = flow / (2 ** self.time_step)
        for _ in range(self.time_step):
            flow = flow + self.warp(image=flow, ddf=flow,  mode='bilinear', p_mode="border")[1]
        return flow
