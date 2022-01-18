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
    datahead = np.stack([datahead]*100,axix = 0)
    datatail = np.stack([datatail]*100,axix = 0)
    data = np.concatenate([datahead,data,datatail],axis = 0)
    for dim in range(data.shape[1]):
        pts1 = data[:,dim]
        
        fft1 = np.fft.fft(pts1)
        
        fft1[cutoff:-cutoff] = 0
        
        recover = np.fft.ifft(fft1)
        
        data[:dim] = recover
    data = data[100:-100]
    return data




def process_img(img, fa, mtcnn, device, args):
    recon_model = get_recon_model(model=args.recon_model, device=device, batch_size=1, img_size=args.tar_size)
    
    img_arr = img[:, :, ::-1]


    resized_face_img = cv2.resize(img_arr, (args.tar_size, args.tar_size))
    
    try:
        lms = fa.get_landmarks_from_image(resized_face_img)[0]
    except:
        return False, 0, 0
    
    lms = lms[:, :2][None, ...]

    return lms

    img_show = resized_face_img.copy()

    # print(lms)
    # for point in lms[0]:
    #     cv2.circle(img_show,(int(point[0]),int(point[1])),2,(255,255,0),1)
    
    # cv2.imshow('img_show',img_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__=='__main__':


    args = ImageFittingOptions()
    args = args.parse()

    device = 'cuda:0'

    mtcnn = MTCNN(select_largest=False, device=device)

    #fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)

    src = 'media/frame_avata_adam'
    dst = 'media/frame_avata_adam_gt_fit'

    lmks = []

    for index in range(len(os.listdir(src))):

        img_name = f'{index:04d}'

        im = cv2.imread(f'{src}/{img_name}.jpg')

        lmk = process_img(im, fa, mtcnn, device, args)

        lmks.append(lmk[0])
    
    print(len(lmks))
    
    pickle.dump(lmks,open(f'{dst}/orginal_lmk.pkl','bw'))
    
    
    