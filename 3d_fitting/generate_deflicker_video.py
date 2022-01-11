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
import pickle
import core.utils as utils
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from inpainter import Inpainter
from inpainter.options import Options
import torchvision.transforms as transforms
import time

def load_target(start,end,path):
    
    mydict={}
    for i in range(start,end-1):
        coeffs = pickle.load(open(f'{path}/{i:04d}_coeffs.pkl','br'))
        crop_img = Image.open(f'{path}/{i:04d}_crop.jpg')  
        
        lmk = pickle.load(open(f'{path}/{i:04d}_lms_proj.pkl','br'))[0]
 
        mydict[f'{i:04d}']=[coeffs,crop_img,lmk]
    return mydict



def process_img(bg, fg, V_writer, bbox,args):
    


    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]

    resized = cv2.resize(fg,(face_w,face_h))
    
    _bg = bg.copy()

    _bg[bbox[1]:bbox[3],bbox[0]:bbox[2]]=resized

    # cv2.imshow('',_bg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    V_writer.write(_bg)

def resample(exp,src_rate,target_rate):
    L,D = exp.shape
    xp = np.arange(0,L/src_rate,1/src_rate).reshape(-1)
    x = np.arange(0,xp[-1],1/target_rate).reshape(-1)


    out = np.zeros([x.shape[0],D], dtype=np.float32)
    for i in range(D):

        buff = np.interp(x,xp,exp[:,i])
        out[:,i] = buff
    return out

if __name__=='__main__':



    args = ImageFittingOptions()
    args = args.parse()

    device = 'cuda:0'

    #face detection
    mtcnn = MTCNN(select_largest=False, device=device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)
    opt = Options()

    #pytorch render
    recon_model = get_recon_model(model=args.recon_model, device=device, batch_size=1, img_size=opt.IMG_size)

    inpainter =Inpainter.Inpainter(opt,opt.model_path)

    out_folder = opt.ouput

    target = opt.target_path

    if opt.src_expression.endswith('.pkl'):
        src_expression = pickle.load(open(f'{opt.src_expression}','br'))

        src_expression = resample(src_expression,60,30)

    else:
        src_expression = [x for x in os.listdir(opt.src_expression) if x.endswith('coeffs.pkl')]
        src_expression = sorted(src_expression)
        #print(src_expression)
        src_expression = [ pickle.load(open(f'{opt.src_expression}/{x}','br'))[:, 80:144]   for x in src_expression]

    src_size = len(src_expression)

    print('experesion length',len(src_expression))

    index_start = 0
    end_index = int(len(os.listdir(target))/4)

    print('target end index',end_index)

    target_info =  load_target(index_start,end_index,target)

    #extract background frames
    background_frames  = {}
    cap  = cv2.VideoCapture(f'{ opt.background_v}')
    frame_cnt = 0
    while 1:
        ret,background = cap.read()
        if not ret:
            break
        
        if opt.cuthead:
            if frame_cnt>19:
                background_frames[f'{frame_cnt-20:04d}']=background
        else:
                background_frames[f'{frame_cnt:04d}']=background
            
        frame_cnt+=1

        if frame_cnt > 2000:
            break

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = opt.FPS
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
    # height = int(512)
    # width = int(512)
    print("fps", fps)
    print("frame_height", height)
    print("frame_width", width)

    V_writer = cv2.VideoWriter(f'{out_folder}/video_deflicker.mp4',fourcc, fps, (width,height))

    id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = recon_model.split_coeffs(target_info[f'{0:04d}'][0])

    previous_trans = translation
    previous_angle = angles
    previous_idcoeff = id_coeff
    previous_tex_coeff = tex_coeff
    previous_exp = src_expression[0]

    pdist = nn.PairwiseDistance(p=2)

    t0 = time.time()
    for i,exp in enumerate(src_expression[:-1]):

        if i > 1500:
            break

        ID = i+index_start

        ID = ID%(len(target_info))


        #resize back
        bg = background_frames[f'{ID:04d}']
        
        print(f'{out_folder}/output/{ID:05d}.jpg')
        fg = cv2.imread(f'{out_folder}/output/{ID:05d}.jpg')
        

        process_img(bg,fg,V_writer,opt.bbox,args)
        
        c = i+1
        t1 = time.time()
        
        # print('time', (t1-t0)/c)
        # print('FPS', 1/((t1 - t0)/c))
        
    V_writer.release()

    
    