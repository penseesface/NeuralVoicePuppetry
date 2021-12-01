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



def process_img(bg, fg, src,threedD,dst_path,ID, mtcnn,V_writer, args):
    
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

    cv2.imwrite(f'{dst_path}/{ID}_final.jpg',_bg)

    # cv2.imshow('',_bg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    #resize render (960,540)
    render = cv2.resize(_bg,(540,960))
    target = cv2.resize(bg,(540,960))
    src = cv2.resize(src,(540,304))

    canvas = np.zeros((960,540*3,3),dtype=np.uint8)


    canvas[:,-540:,:]=target
    canvas[:,540:540*2,:]=render
    canvas[:304,:540,:]=src
    canvas[304:304+256,:256,:]=threedD

    V_writer.write(canvas)
    


if __name__=='__main__':


    args = ImageFittingOptions()
    args = args.parse()

    device = 'cuda:0'

    mtcnn = MTCNN(select_largest=False, device=device)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)

    rerendered = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/transfer_render'

    background = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/frames_jane'

    src_folder= '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/frame_src'

    threeD_render = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/transfered/test'

    final = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/final_compose'

    frames_ID = sorted([x.split('_')[0] for x in os.listdir(rerendered)])



    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 25
    height = 960
    width = 540*3
    print("fps", fps)
    print("frame_height", height)
    print("frame_width", width)

    V_writer = cv2.VideoWriter(f'videorendered_finalcompose.mp4',fourcc, fps, (width,height))


    for i,ID in enumerate(frames_ID):

        fg = cv2.imread(f'{rerendered}/{ID}_render.jpg')

        bg = cv2.imread(f'{background}/{ID}.jpg')

        src = cv2.imread(f'{src_folder}/{i:04d}.jpg')

        print(f'{src_folder}/{i:04d}.jpg')

        threedD = cv2.imread(f'{threeD_render}/{ID}_render.jpg')


        process_img(bg,fg,src,threedD,final,ID,mtcnn,V_writer,args)

    V_writer.release()
        