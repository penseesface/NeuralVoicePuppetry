import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from facenet_pytorch import MTCNN
from core.options import ImageFittingOptions
import cv2
import face_alignment
import numpy as np
from core import get_recon_model
import os
import torch
import core.utils as utils
from tqdm import tqdm
import core.losses as losses
from scipy.io import loadmat
import pickle
import multiprocessing as mp
import glob 
import time 
import matplotlib.pyplot as plt
from PIL import Image

def load_target(start,end,path):
    
    mydict={}
    for i in range(start,end-1):
        print(i)
        coeffs = pickle.load(open(f'{path}/train/{i:04d}_coeffs.pkl','br'))
        crop_img = Image.open(f'{path}/train/{i:04d}_crop.jpg')    
        mydict[f'{i:04d}']=[coeffs,crop_img]
    return mydict

if __name__=='__main__':


    args = ImageFittingOptions()
    args = args.parse()

    device = 'cuda:0'


    # mtcnn = MTCNN(select_largest=False, device=device)

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)


    recon_model = get_recon_model(model=args.recon_model, device=device, batch_size=1, img_size=args.tar_size)

    src = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/expression_src'

    target = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/frames_jane_render'

    transfered = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/transfered'

    src_expression_path = sorted( os.listdir(src))

    src_expression = [pickle.load(open(f'{src}/{x}','br')) for x in src_expression_path]

    src_size = len(src_expression)

    index_start = 81

    target_info =  load_target(index_start,src_size+index_start,target)


    for i,exp in enumerate(src_expression[:-1]):

        target_coeffs = target_info[f'{i+index_start:04d}'][0] 
        
        print('target',target_coeffs.shape)

        target_img = target_info[f'{i+index_start:04d}'][1] 

        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = recon_model.split_coeffs(target_coeffs)


        new_coeffes = recon_model.merge_coeffs( id_coeff.cuda(), torch.Tensor(exp).cuda(), tex_coeff.cuda(), angles.cuda(), gamma.cuda(), translation.cuda() )


        result = recon_model(new_coeffes)

        img = result['rendered_img']

        print(img.shape)

        im = Image.fromarray(img[0,...,:3].cpu().numpy().astype(np.uint8))

        im.save(f'{transfered}/{i+index_start:04d}_render.jpg')
        
        target_img.save(f'{transfered}/{i+index_start:04d}_crop.jpg')