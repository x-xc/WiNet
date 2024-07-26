import os
import pytorch_lightning as pl
from nn_util import *
from loss import *
from eval import save_flow, jacobian_determinant_vxm

loss_smooth = smoothloss

class BaseModel(pl.LightningModule):
    def __init__(self, args, model, som):
        super().__init__()
        self.args = args
        self.som = som
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.stn   = SpatialTransform()
        self.diff  = None

    def forward(self, m, f, optimizer_idx=None):
        flow = self.model(m, f)
        return flow
    
    def get_inputs(self, batch):
        if len(batch) == 2:
            m, f = batch
            m_lab, f_lab = None, None
        if len(batch) == 4:
            m, f, m_lab, f_lab = batch
        if len(batch) == 6:
            m, f, m_lab, f_lab = batch[2:]
        return m, f, m_lab, f_lab
    
    def training_step(self, batch, batch_idx):
        steps = self.trainer.global_step 
        
        m,f,_,_ = self.get_inputs(batch)
        ddf = self(m.float(), f.float())
        diff_ddf = self.diff(ddf) if self.args.diff else ddf
        grid, warped = self.stn(m, diff_ddf.movedim(1, -1))
        loss_sim = self.loss_similarity(f, warped) 
        loss_reg = loss_smooth(ddf) * self.som
        loss = loss_sim + loss_reg
        
        self.log("train/loss_sim", loss_sim.mean().detach(), logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_som", loss_reg.mean().detach(), logger=True, on_step=True, on_epoch=True)
        self.log("train/loss",     loss.mean().detach(),     logger=True, on_step=True, on_epoch=True)
        
        is_log_img = steps % 1000 == 0
        if is_log_img:
            steps = self.trainer.global_step
            path = os.path.join(self.logger.log_dir, 'img_logs')
            os.makedirs(path, exist_ok=True)
            g, w = grid, warped
            save_flow(m[:,:,:,:,112], f[:,:,:,:,112], w[:,:,:,:,112], g[:,:,:,112,1:].permute(0, 3, 1, 2), path+f'/sample_{steps}.jpg')          
        return loss
    
    def validation_step(self, batch, batch_idx):
        m, f, m_lab, f_lab = self.get_inputs(batch)
        ddf = self(m.float(), f.float())
        diff_ddf = self.diff(ddf) if self.args.diff else ddf
        _, warped = self.stn(m, diff_ddf.movedim(1, -1))
        loss_sim =  self.loss_similarity(f, warped)
        g, warped_lab = self.stn(m_lab.float(), diff_ddf.movedim(1, -1), mod ='nearest')
       
        dice_score = self.args.dice_val(warped_lab.long(), f_lab.long())
        self.log("val/loss_sim",   loss_sim,   logger=True,  on_step=True)
        self.log("val/dice_score", dice_score, logger= True, on_step=True)
        return loss_sim, dice_score
    
    def test_step(self, batch, batch_idx):
        return None
        

    def configure_optimizers(self):
        lr = self.args.lr
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
    
        return [opt], []