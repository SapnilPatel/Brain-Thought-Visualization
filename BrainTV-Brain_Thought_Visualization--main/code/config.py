"""
Author: Neel Shah, Sapnil Patel, Yagnik Poshiya, Zijiao, Jack Qing, Patrick Finley
GitHub:  @neeldevenshah, @SapnilPatel, @yagnikposhiya, @zjc062, @jqin4749, @patrick-finley
Team name: ThreeMinds
Charotar University of Science and Technology

@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Zijiao and Qing, Jiaxin and Xiang, Tiange and Yue, Wan Lin and Zhou, Juan Helen},
    title     = {Seeing Beyond the Brain: Masked Modeling Conditioned Diffusion Model for Human Vision Decoding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
"""

import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

# pass keyword is placeholder, used to define empty class

class Config_MBM_fMRI(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 50 # NOTE: changed
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8

        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 16
        self.embed_dim = 1024 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0 # TODO: explore mlp_ratio

        # Project setting
        self.root_path = '/scratch/three-minds/workstation/lab/LatestBrain-TV/'
        self.output_path = self.root_path
        self.seed = 2022
        self.roi = 'VC' #TODO: explore 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = False # NOTE: changed
        self.accum_iter = 1 # TODO: explore accum_iter

        self.use_nature_img_loss = False # TODO: what is nature_img_loss?
        self.img_recon_weight = 0.5 # TODO: for what img_recon_weight is used?
        self.focus_range = None # [0, 1500] # None to disable it #TODO: focus range of what?
        self.focus_rate = 0.6 # TODO: focus rate of what?

        # distributed training
        self.local_rank = 0
        # if distributed training is enable then machine which executes code & has '0' index will be behaved as a head machine.

class Config_MBM_finetune(Config_MBM_finetune):
    def __init__(self):
        
        # Project setting
        self.root_path = '/scratch/three-minds/workstation/lab/LatestBrain-TV/'
        self.output_path = self.root_path
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.dataset = 'GOD' # GOD  or BOLD5000
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth') 

        self.include_nonavg_test = True # TODO: explore include_nonavg_test keyword
        self.kam_subs = ['sbj_3'] # take only 1 subject data from kamitani
        self.bold5000_subs = ['CSI1','CSI2','CSI3','CSI4'] # take all subject data from BOLD5000

        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.75 
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.

        # distributed training
        self.local_rank = 0

class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '/scratch/three-minds/workstation/lab/LatestBrain-TV/'
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.roi = 'VC'
        self.patch_size = 16

        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/semantic')
        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/label2img')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/text2img-large')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/layout2img')

        self.dataset = 'GOD' # GOD or BOLD5000
        self.kam_subs = ['sbj_3']
        self.bold5000_subs = ['CSI1']
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth')

        self.img_size = 256

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 500
        
        self.precision = 32 # TODO: explore, what is precision here, parameter or performance evaluator?
        self.accumulate_grad = 1 # TODO: 
        self.crop_ratio = 0.2
        self.global_pool = False # TODO: explore global_pool
        self.use_time_cond = True # TODO: explore use_time_cond
        self.eval_avg = True # TODO: explore eval_avg?

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250 # TODO: explore ddim_samples; The DDIM Stable Diffusion method is an extension of the k-LMS Stable Diffusion algorithm and provides more precise sampling.
        self.HW = None # TODO: explore HW
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None # os.path.join(self.root_path, 'results/generation/25-08-2022-08:02:55/checkpoint.pth')
