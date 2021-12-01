import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from PIL import Image


def make_dataset(dir):
    images_render = []
    images_crop = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['render.jpg']):
                id_str = fname.split('_')[0]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)

    for id in ids:
        fname=f'{id:04d}_render.jpg'
        path = os.path.join(root, fname)
        images_render.append(path)
    

    for id in ids:
        fname=f'{id:04d}_crop.jpg'
        path = os.path.join(root, fname)
        images_crop.append(path)
    

    return images_render,images_crop,ids


class Aligneddataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.data_dir = os.path.join(opt.dataroot, opt.phase)

        self.render_paths,self.crop_paths, self.ids  = make_dataset(self.data_dir)

        opt.nObjects = 1
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):

        # get video data
        frame_id = index

        #print('GET ITEM: ', index)
        render_path = self.render_paths[index]

        crop_path = self.crop_paths[index]

        # default image dimensions
        IMG_DIM_X = 256
        IMG_DIM_Y = 256

        # load image data
 
        img_array_render = np.asarray(Image.open(render_path))/255
        img_array_crop = np.asarray(Image.open(crop_path))/255


        TARGET = transforms.ToTensor()(img_array_crop.astype(np.float32))
        render = transforms.ToTensor()(img_array_render.astype(np.float32))

        TARGET = 2.0 * TARGET - 1.0
        render = 2.0 * render - 1.0


        ID = self.ids[index]
        #################################
        ####### apply augmentation ######
        #################################
        if not self.opt.no_augmentation:
            # random dimensions
            new_dim_x = np.random.randint(int(IMG_DIM_X * 0.75), IMG_DIM_X+1)
            new_dim_y = np.random.randint(int(IMG_DIM_Y * 0.75), IMG_DIM_Y+1)
            new_dim_x = int(np.floor(new_dim_x / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
            new_dim_y = int(np.floor(new_dim_y / 64.0) * 64 )
            if new_dim_x > IMG_DIM_X: new_dim_x -= 64
            if new_dim_y > IMG_DIM_Y: new_dim_y -= 64

            # random pos
            if IMG_DIM_X == new_dim_x: offset_x = 0
            else: offset_x = np.random.randint(0, IMG_DIM_X-new_dim_x)
            if IMG_DIM_Y == new_dim_y: offset_y = 0
            else: offset_y = np.random.randint(0, IMG_DIM_Y-new_dim_y)

            # select subwindow
            # TARGET = TARGET[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
            # render = render[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]

            # compute new intrinsics
            # TODO: atm not needed but maybe later


        #################################

        return {'TARGET': TARGET, 'rendered': render,'ID':ID}

    def __len__(self):


        return len(self.render_paths)

    def name(self):
        return 'Aligneddataset'
