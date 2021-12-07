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

def load_target(start,end,path):
    
    mydict={}
    for i in range(start,end-1):
        print(i)
        coeffs = pickle.load(open(f'{path}/train/{i:04d}_coeffs.pkl','br'))
        crop_img = Image.open(f'{path}/train/{i:04d}_crop.jpg')  
        
        lmk = pickle.load(open(f'{path}/train/{i:04d}_lms_proj.pkl','br'))[0]
 
        mydict[f'{i:04d}']=[coeffs,crop_img,lmk]
    return mydict



def process_img(bg, fg, mtcnn, frame, V_writer, args):
    
    img_arr = bg[:, :, ::-1]
    orig_h, orig_w = img_arr.shape[:2]
    bboxes, probs = mtcnn.detect(img_arr)
    if bboxes is None:
        return None
    if len(bboxes) == 0:
        return None
    bbox = utils.pad_bbox(bboxes[0], (orig_w, orig_h), args.padding_ratio)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]

    resized = cv2.resize(fg,(face_w,face_h))


    _bg = bg.copy()

    _bg[bbox[1]:bbox[3],bbox[0]:bbox[2]]=resized


    #resize render (960,540)
    render = cv2.resize(_bg,(1080,1920))
    target = cv2.resize(bg,(540,960))

    src = cv2.resize(frame,(540,311))

    canvas = np.zeros((1920,1080,3),dtype=np.uint8)

    print(canvas.shape)
    #canvas[:,-540:,:]=target
    canvas[:,:,:]=render
    canvas[:311,:540,:]=src

    V_writer.write(canvas)



if __name__=='__main__':



    args = ImageFittingOptions()
    args = args.parse()

    device = 'cuda:0'

    #face detection
    mtcnn = MTCNN(select_largest=False, device=device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)

    #pytorch render
    recon_model = get_recon_model(model=args.recon_model, device=device, batch_size=1, img_size=args.tar_size)

    opt = Options()
    inpainter =Inpainter.Inpainter(opt)

    out_folder = 'media/2nd_version_chinese'

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(f'{out_folder}/debug'):
        os.mkdir(f'{out_folder}/debug')

    target = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/frames_jane_render2'

    background = 'media/frames_jane'

    src_expression = pickle.load(open(f'media/predicted_expression_xinwenlianbo.pkl','br'))
    

    src_size = len(src_expression)

    print(len(src_expression))
    index_start = 300

    target_info =  load_target(index_start,src_size+index_start,target)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    height = 1920
    width = 1080
    print("fps", fps)
    print("frame_height", height)
    print("frame_width", width)


    V_writer = cv2.VideoWriter(f'{out_folder}/videorendered.mp4',fourcc, fps, (width,height))

    cap  = cv2.VideoCapture('media/1048.mp4')

    for i,exp in enumerate(src_expression[:-1]):

        ret,frame = cap.read()

        if not ret:
            break

        ID = i+index_start

        target_coeffs = target_info[f'{ID:04d}'][0] 
        
        # render 3D face
        target_img = target_info[f'{ID:04d}'][1] 
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = recon_model.split_coeffs(target_coeffs)
        new_coeffes = recon_model.merge_coeffs( id_coeff.cuda(), torch.Tensor(exp).cuda().view(1,64), tex_coeff.cuda(), angles.cuda(), gamma.cuda(), translation.cuda() )
        result = recon_model(new_coeffes)

        #load landmark

        landmark = target_info[f'{ID:04d}'][2] 
        lmk_index = [2,3,4,5,6,7,8,9,10,11,12,13,14,29]
        landmark_select = landmark[lmk_index]
        mask = np.zeros((256,256,3))
        pts = landmark_select.reshape((-1,1,2))
        pts = np.array(pts,dtype=np.int32)
        mask = cv2.fillPoly(mask,[pts],(255,255,255))
        mask = transforms.ToTensor()(mask.astype(np.float32))

        # norm 
        render = (result['rendered_img'] / 255 * 2 -1)[0,:,:,:3]
        render =  render.permute(2, 0, 1)
        img_array_crop = np.asarray(target_img)/255
        TARGET = transforms.ToTensor()(img_array_crop.astype(np.float32))
        TARGET = 2.0 * TARGET - 1.0


        #rerender crop face
        fake = inpainter(TARGET,render,mask)
        fg = Inpainter.tensor2im(fake.clone())
        fg = fg[:,:,::-1]


        #debug
        _render_copy = ((render.permute(1,2,0)+1)/2*255).cpu().numpy()[:,:,::-1]
        _render_copy = _render_copy.astype(np.uint8)
        saved = np.ones((256,256*2,3),dtype=np.uint8)
        saved[:,:256,:] = _render_copy
        saved[:,256:,:] = fg

        cv2.imwrite(f'{out_folder}/debug/{ID:04d}.jpg',saved)

        # print(fg.shape)
        # cv2.imshow('render',_render_copy) 
        # cv2.imshow('fg',saved)
        # cv2.waitKey() 
        # cv2.destroyAllWindows()



        #resize back
        bg = cv2.imread(f'{background}/{ID:04d}.jpg')

        process_img(bg,fg,mtcnn,frame,V_writer,args)

    V_writer.release()
        