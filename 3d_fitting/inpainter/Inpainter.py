import torch
from . import networks
from . options import Options
from . UNET import UnetRenderer
# from UNET import UnetRenderer
# import networks
# from options import Options
import os
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pickle as pkl
import cv2

def define_Inpainter(renderer, n_feature, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm)
    N_OUT = 3
    net = UnetRenderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return networks.init_net(net, init_type, init_gain, gpu_ids)

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        input_image = torch.clamp(input_image, -1.0, 1.0) 
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


INVALID_UV = 0

class Inpainter(nn.Module):
    def __init__(self,opt,pretrained='3d_fitting/inpainter/2nd_45_inpainter_ymh.pth') -> None:
        super(Inpainter, self).__init__()

        self.device = opt.device

        self.inpainter = define_Inpainter(opt.rendererTypes, 6, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

        self._load_networks(f'{pretrained}')

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def _load_networks(self, load_path):
        net = self.inpainter
        print('loading the module from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)



    def forward(self,TARGET,render,mask):
        self.target = torch.unsqueeze( TARGET.to(self.device),dim = 0)
        self.rendered = torch.unsqueeze(render.to(self.device),dim =0)          
        self.mask = torch.unsqueeze(mask.to(self.device),dim =0)          

        # background        
        # mask = self.rendered[:,0:1,:,:]==INVALID_UV

        # mask = torch.cat([mask,mask,mask], 1)

        # self.background = torch.where(mask, self.target, torch.zeros_like(self.target))


        mask = self.mask == INVALID_UV
        #mask = mask.permute(0,3,1,2)

        self.background = torch.where(mask, self.target, torch.zeros_like(self.target))

        self.rendered = torch.where(mask, torch.zeros_like(self.target), self.rendered)


        self.fake = self.inpainter(self.rendered, self.background)

        #self.fake = torch.cat(self.fake, dim=0)


        self.fake = torch.where(mask, self.background, self.fake)

        return self.fake

if __name__ == '__main__':
    opt = Options()
    compose_model = Inpainter(opt)

    img_array_render = np.asarray(Image.open('3d_fitting/inpainter/0082_render.jpg'))/255
    img_array_crop = np.asarray(Image.open('3d_fitting/inpainter/0082_crop.jpg'))/255



    landmark = pkl.load(open('3d_fitting/inpainter/0082_lms_proj.pkl','rb'))[0]
    lmk_index = [2,3,4,5,6,7,8,9,10,11,12,13,14,29]


    landmark_select = landmark[lmk_index]

    mask = np.zeros((256,256,3))

    pts = landmark_select.reshape((-1,1,2))

    pts = np.array(pts,dtype=np.int32)

    mask = cv2.fillPoly(mask,[pts],(255,255,255))

    TARGET = transforms.ToTensor()(img_array_crop.astype(np.float32))
    render = transforms.ToTensor()(img_array_render.astype(np.float32))
    mask = transforms.ToTensor()(mask.astype(np.float32))

    TARGET = 2.0 * TARGET - 1.0
    render = 2.0 * render - 1.0


    fake = compose_model(TARGET,render,mask)

    result =tensor2im(fake.clone())

    Image.fromarray(result).save(f'3d_fitting/inpainter/test.jpg')
