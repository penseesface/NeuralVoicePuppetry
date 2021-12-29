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

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(f'{out_folder}/debug'):
        os.mkdir(f'{out_folder}/debug')

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
        
        if frame_cnt>19:
            background_frames[f'{frame_cnt-20:04d}']=background
        
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

    V_writer = cv2.VideoWriter(f'{out_folder}/videorendered.mp4',fourcc, fps, (width,height))


    id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = recon_model.split_coeffs(target_info[f'{0:04d}'][0])

    previous_trans = translation
    previous_angle = angles
    previous_idcoeff = id_coeff
    previous_tex_coeff = tex_coeff
    previous_exp = src_expression[0]


    t0 = time.time()
    for i,exp in enumerate(src_expression[:-1]):

        if i > 2000:
            break

        ID = i+index_start

        ID = ID%(len(target_info))

        target_coeffs = target_info[f'{ID:04d}'][0] 
        
        # render 3D face
        target_img = target_info[f'{ID:04d}'][1] 
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = recon_model.split_coeffs(target_coeffs)
        
        new_translation = opt.mvg_lamda*previous_trans+(1-opt.mvg_lamda)*translation 
        new_angles = opt.mvg_lamda*previous_angle+(1-opt.mvg_lamda)*angles 

        new_exp = opt.src_exp_lamda*previous_exp+(1-opt.src_exp_lamda)*exp 
        if i>0: 
            previous_trans = new_translation
            previous_angle = new_angles
            previous_exp = new_exp

        new_coeffes = recon_model.merge_coeffs( previous_idcoeff.cuda(), torch.Tensor(exp).cuda().view(1,64), tex_coeff.cuda(), new_angles.cuda(), gamma.cuda(), new_translation.cuda() )
        result = recon_model(new_coeffes)

        #load landmark
        landmark = target_info[f'{ID:04d}'][2] 
        lmk_index = [2,3,4,5,6,7,8,9,10,11,12,13,14,29]
        landmark_select = landmark[lmk_index]
        mask = np.zeros((opt.IMG_size,opt.IMG_size,3))
        pts = landmark_select.reshape((-1,1,2))
        pts = np.array(pts,dtype=np.int32)
        mask = cv2.fillPoly(mask,[pts],(255,255,255))

        kernal = np.ones((3,3),np.uint)
        mask = cv2.dilate(mask,kernel=kernal,iterations=4)
        

        mask = transforms.ToTensor()(mask.astype(np.float32))

        # norm 
        render = (result['rendered_img'] / 255 * 2 -1)[0,:,:,:3]
        render =  render.permute(2, 0, 1)
        img_array_crop = np.asarray(target_img)/255
        TARGET = transforms.ToTensor()(img_array_crop.astype(np.float32))
        TARGET = 2.0 * TARGET - 1.0

        #rerender crop face
        # print('mask shape',mask.shape)
        # print('Target shape',TARGET.shape)
        # print('render shape',render.shape)
        fake = inpainter(TARGET,render,mask)
        fg = Inpainter.tensor2im(fake.clone())
        fg = fg[:,:,::-1]


        #debug
        _render_copy = ((render.permute(1,2,0)+1)/2*255).cpu().numpy()[:,:,::-1]
        _render_copy = _render_copy.astype(np.uint8)
        saved = np.ones((opt.IMG_size,opt.IMG_size*2,3),dtype=np.uint8)
        saved[:,:opt.IMG_size,:] = _render_copy
        saved[:,opt.IMG_size:,:] = fg

        cv2.imwrite(f'{out_folder}/debug/{ID:04d}.jpg',saved)

        # print(fg.shape)
        # cv2.imshow('render',_render_copy) 
        # cv2.imshow('fg',saved)
        # cv2.waitKey() 
        # cv2.destroyAllWindows()

        #resize back
        bg = background_frames[f'{ID:04d}']

        process_img(bg,fg,V_writer,opt.bbox,args)
        c = i+1
        t1 = time.time()
        # print('time', (t1-t0)/c)
        # print('FPS', 1/((t1 - t0)/c))
        
    V_writer.release()
        