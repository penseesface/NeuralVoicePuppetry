import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import functools
from PIL import Image
from torchvision import models
from collections import namedtuple


from . import VGG_LOSS
from . import UNET

from util import util

from util.image_pool import ImagePool
import cv2
################
###  HELPER  ###
################
INVALID_UV = -1.0

from models import networks
from models.base_model import BaseModel


def define_Inpainter(renderer, n_feature, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm)
    N_OUT = 3
    #renderer=='UNET_5_level'
    net = UNET.UnetRenderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    return networks.init_net(net, init_type, init_gain, gpu_ids)



class piplinemodel(BaseModel):
    def name(self):
        return 'DynamicNeuralTexturesModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        #parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(norm='instance', netG='unet_256')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.trainRenderer = not opt.fix_renderer

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_total', 'G_L1_Rendering', 'G_VGG_Rendering', 'G_GAN']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['rendered', 'fake', 'target']
        else:
            self.visual_names = ['rendered', 'fake', 'target']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = [ 'inpainter' ,'netD']
        else:  # during test time, only load Gs
            self.model_names = ['inpainter']


        # load/define networks
        self.inpainter = define_Inpainter(opt.rendererType, 6, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        # optimizer
        self.loss_G_GAN = 0.0

        print(self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc , opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='mean')
            self.criterionL1Smooth = torch.nn.SmoothL1Loss(reduction='mean')
            self.criterionL2 = torch.nn.MSELoss(reduction='mean')

            #if self.opt.lossType == 'VGG':
            self.vggloss = VGG_LOSS.VGGLOSS().to(self.device)

            # initialize optimizers
            self.optimizers = []

            self.optimizer_inpainter = torch.optim.Adam(self.inpainter.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_inpainter)

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)



    def set_input(self, input):
        self.target = input['TARGET'].to(self.device)
        self.rendered = input['rendered'].to(self.device)             


    def forward(self, alpha=1.0):
        # background        
        mask = self.rendered[:,0:1,:,:]==INVALID_UV

        mask = torch.cat([mask,mask,mask], 1)

        self.background = torch.where(mask, self.target, torch.zeros_like(self.target))


        result =util.tensor2im(self.background.clone())

        # Image.fromarray(result).save(f'media/background/bg.jpg')

        self.fake = self.inpainter(self.rendered, self.background)

        #self.fake = torch.cat(self.fake, dim=0)


        self.fake = torch.where(mask, self.background, self.fake)


    def backward_D(self):
        mask = self.rendered[:,0:1,:,:] != INVALID_UV
        mask = torch.cat([mask,mask,mask], 1)
        def masked(img):
            return torch.where(mask, img, torch.zeros_like(img))

        # Fake
        # stop backprop to the generator by detaching fake_B
        # fake_AB = self.fake_AB_pool.query(torch.cat((self.rendered, masked(self.fake)), 1))
        fake_AB = self.fake_AB_pool.query(masked(self.fake))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        # real_AB = torch.cat((self.input_uv, masked(self.target)), 1)
        real_AB =  masked(self.target)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self, epoch):

        mask = self.rendered[:,0:1,:,:] != INVALID_UV
        sum_mask = torch.sum(mask)
        d = mask.shape[1]
        mask_weight = (d*d) / sum_mask
        mask = torch.cat([mask,mask,mask], 1)
        def masked(img):
            return torch.where(mask, img, torch.zeros_like(img))

        # First, G(A) should fake the discriminator
        fake_AB = masked(self.fake)
        pred_fake = self.netD(fake_AB)
        
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.0 # disabled GAN


        # Second, G(A) = B
        self.loss_G_L1_Rendering = 0.0

        self.loss_G_L1_Rendering += 10.0 * self.criterionL1(self.fake, self.target)

        self.loss_G_VGG_Rendering = 0.0

        #if self.opt.lossType == 'VGG':
            
        self.loss_G_VGG_Rendering += 10.0 * self.vggloss(self.fake, self.target)

        self.loss_G_total = self.loss_G_L1_Rendering + self.loss_G_VGG_Rendering + self.loss_G_GAN

        self.loss_G_total.backward()

    def optimize_parameters(self, epoch_iter):
        alpha = (epoch_iter-5) / 50.0
        if alpha < 0.0: alpha = 0.0
        if alpha > 1.0: alpha = 1.0
        self.forward(alpha)


        updateDiscriminator = self.loss_G_GAN < 1.0#0.1

        # update Discriminator
        if updateDiscriminator:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update Generator
        self.set_requires_grad(self.netD, False)
        self.optimizer_inpainter.zero_grad()

        self.backward_G(epoch_iter)

        self.optimizer_inpainter.step()

