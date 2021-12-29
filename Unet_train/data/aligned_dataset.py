import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from PIL import Image
import pickle as pkl
import cv2

def make_dataset(dir):
    images_render = []
    images_crop = []
    landmarks = []
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
    

    for id in ids:
        fname=f'{id:04d}_lms_proj.pkl'
        path = os.path.join(root, fname)
        landmarks.append(path)
    

    return images_render,images_crop,ids,landmarks


class Aligneddataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.data_dir = os.path.join(opt.dataroot, opt.phase)

        self.render_paths,self.crop_paths, self.ids, self.landmarks  = make_dataset(self.data_dir)

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

        landmark = pkl.load(open(self.landmarks[index],'rb'))[0]

        lmk_index = [2,3,4,5,6,7,8,9,10,11,12,13,14,29]

        landmark_select = landmark[lmk_index]

        mask = np.zeros((IMG_DIM_X,IMG_DIM_Y,3))

        pts = landmark_select.reshape((-1,1,2))

        pts = np.array(pts,dtype=np.int32)

        mask = cv2.fillPoly(mask,[pts],(255,255,255))

        itera = np.random.randint(5)
        kernal = np.ones((3,3),np.uint)

        if np.random.rand() > 0.3:
            mask = cv2.dilate(mask,kernel=kernal,iterations=itera)
        
        
        if np.random.rand() > 0.7:
            mask = cv2.erode(mask,kernel=kernal,iterations=itera)

        # cv2.imshow('dilate',mask_dilate)
        # cv2.imshow('mask',mask)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        TARGET = transforms.ToTensor()(img_array_crop.astype(np.float32))
        render = transforms.ToTensor()(img_array_render.astype(np.float32))
        mask = transforms.ToTensor()(mask.astype(np.float32))

        TARGET = 2.0 * TARGET - 1.0
        render = 2.0 * render - 1.0


        ID = self.ids[index]
        #################################
        ####### apply augmentation ######
        #################################


        #################################

        return {'TARGET': TARGET, 'rendered': render,'ID':ID,'mask':mask}

    def __len__(self):


        return len(self.render_paths)

    def name(self):
        return 'Aligneddataset'
