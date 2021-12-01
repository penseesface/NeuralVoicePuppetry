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

def process_img(img, fa, mtcnn, device, args):
    recon_model = get_recon_model(model=args.recon_model, device=device, batch_size=1, img_size=args.tar_size)
    
    img_arr = img[:, :, ::-1]
    orig_h, orig_w = img_arr.shape[:2]
    bboxes, probs = mtcnn.detect(img_arr)
    if bboxes is None:
        return False, 0, 0
    if len(bboxes) == 0:
        return False, 0, 0
    bbox = utils.pad_bbox(bboxes[0], (orig_w, orig_h), args.padding_ratio)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    face_img = img_arr[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
    resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))
    try:
        lms = fa.get_landmarks_from_image(resized_face_img)[0]
    except:
        return False, 0, 0
    lms = lms[:, :2][None, ...]
    lms = torch.tensor(lms, dtype=torch.float32, device=device)
    img_tensor = torch.tensor(
        resized_face_img[None, ...], dtype=torch.float32, device=device)

    lm_weights = utils.get_lm_weights(device)
    rigid_optimizer = torch.optim.Adam([recon_model.get_rot_tensor(), recon_model.get_trans_tensor()], lr=args.rf_lr)
    for i in range(args.first_rf_iters):
        rigid_optimizer.zero_grad()
        pred_dict = recon_model(recon_model.get_packed_tensors(), render=False)
        lm_loss_val = losses.lm_loss(pred_dict["lms_proj"], lms, lm_weights, img_size=args.tar_size)
        total_loss = args.lm_loss_w * lm_loss_val
        total_loss.backward()
        rigid_optimizer.step()

    nonrigid_optimizer = torch.optim.Adam(
        [recon_model.get_id_tensor(), recon_model.get_exp_tensor(), recon_model.get_gamma_tensor(), recon_model.get_tex_tensor(), recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
        lr=args.nrf_lr,
    )
    for i in range(args.first_nrf_iters):
        nonrigid_optimizer.zero_grad()

        pred_dict = recon_model(recon_model.get_packed_tensors(), render=True)
        rendered_img = pred_dict['rendered_img']
        lms_proj = pred_dict['lms_proj']
        face_texture = pred_dict['face_texture']

        mask = rendered_img[:, :, :, 3].detach()

        photo_loss_val = losses.photo_loss(
            rendered_img[:, :, :, :3], img_tensor, mask > 0)

        lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights,
                                     img_size=args.tar_size)
        id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
        exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())
        tex_reg_loss = losses.get_l2(recon_model.get_tex_tensor())
        tex_loss_val = losses.reflectance_loss(
            face_texture, recon_model.get_skinmask())

        loss = lm_loss_val*args.lm_loss_w + \
            id_reg_loss*args.id_reg_w + \
            exp_reg_loss*args.exp_reg_w + \
            tex_reg_loss*args.tex_reg_w + \
            tex_loss_val*args.tex_w + \
            photo_loss_val*args.rgb_loss_w

        loss.backward()
        nonrigid_optimizer.step()

    coeffs = recon_model.get_packed_tensors()
    id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = recon_model.split_coeffs(coeffs)
    # vs = recon_model.get_vs(id_coeff, exp_coeff)
    # pickle.dump([exp_coeff.cpu().detach().numpy(), angles.cpu().detach().numpy(), translation.cpu().detach().numpy()], open('abc.pkl', 'wb'))
    return True, exp_coeff.cpu().detach().numpy(), lms.cpu().detach().numpy()

def process_video(Q, device, args):
    while Q.qsize()>0:
        frame, i, video_name = Q.get()
        print(video_name, i, device, Q.qsize())
        target_name = './video_results/%s/%06d.pkl'%(video_name.split('/')[-1].replace('.mov', ''), i)
        os.makedirs('./video_results/%s/'%(video_name.split('/')[-1].replace('.mov', '')), exist_ok=True)
        if os.path.exists(target_name):
            continue

        mtcnn = MTCNN(select_largest=False, device=device)
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)

        has_face, exp, lmk = process_img(frame, fa, mtcnn, device, args)
        # print(has_face)
        if has_face:
            pickle.dump(exp, open(target_name, 'wb'))
    print(device, 'ended.')


def load_video(video_names, Q):
    for video_name in video_names:
        print('start loading video', video_name)
        cap = cv2.VideoCapture(video_name)
        fps = cap.get(5)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(length):
            # print(i)
            ret, frame = cap.read()
            if ret:
                Q.put([frame, i, video_name])
            else:
                break
    print('Video read finished.')




if __name__=='__main__':
    args = ImageFittingOptions()
    args = args.parse()
    video_names = glob.glob('./videos/*.mp4')
    video_names = sorted(video_names)

    q = mp.Queue(maxsize=100)

    read_p = mp.Process(target=load_video, args=(video_names, q))
    read_p.start()
    # read_p.join()

    time.sleep(4)

    for i in range(8):
        device = 'cuda:%d'%i
        print('process %d'%i, device)
        p = mp.Process(target=process_video, args=(q, device, args))
        p.start()
        # p.join()
