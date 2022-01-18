from curses import window
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

def low_pass(data,cutoff):
    assert cutoff<data.shape[0]//2
    data = data.copy()
    datahead = data[0]
    datatail = data[-1]
    datahead = np.stack([datahead]*100,axis = 0)
    datatail = np.stack([datatail]*100,axis = 0)
    data = np.concatenate([datahead,data,datatail],axis = 0)
    for dim in range(data.shape[1]):
        pts1 = data[:,dim]
        
        fft1 = np.fft.fft(pts1)
        
        fft1[cutoff:-cutoff] = 0
        
        recover = np.fft.ifft(fft1)
        
        data[:,dim] = recover
    data = data[100:-100]
    return data



if __name__=='__main__':


    args = ImageFittingOptions()
    args = args.parse()

    device = 'cuda:0'






    src = 'media/frame_avata_adam'
    dst = 'media/frame_avata_adam_gt_fit'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    height = int(256)
    width = int(512)    


    V_writer = cv2.VideoWriter(f'{dst}/landmark.mp4',fourcc, fps, (width,height))


    lmks = pickle.load(open(f'{dst}/orginal_lmk.pkl','br'))
    
    lmks = np.array(lmks)    
    
    lmk_select = lmks.reshape(lmks.shape[0],136)
    
    lmk_low_passed = low_pass(lmk_select,600)

    lmk_low_passed = lmk_low_passed.reshape(lmk_select.shape[0],68,2)
    
    pickle.dump(lmk_low_passed,open(f'{dst}/lowpass_lmk.pkl','bw'))
    
    for i,lm in enumerate(lmk_low_passed[:1000]):
    
        
        img_name = f'{i:04d}'

        im = cv2.imread(f'{src}/{img_name}.jpg')
        im2 = im.copy()

        for point in lmks[i]:
            cv2.circle(im,(int(point[0]),int(point[1])),2,(255,255,0),1)
        
        for point in lmk_low_passed[i]:
            cv2.circle(im2,(int(point[0]),int(point[1])),2,(255,255,0),1)
        
        saved = np.ones((256,512,3),dtype=np.uint8)

        saved[:,:256,:] = im
        saved[:,256:,:] = im2
        
        V_writer.write(saved)
        # cv2.imshow('img_show',im)
        # cv2.imshow('img_show2',im2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    V_writer.release()