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


    vs = recon_model.get_vs(id_coeff, exp_coeff)

    render_result = recon_model(coeffs)


    return True, render_result['rendered_img'].cpu().detach().numpy(),resized_face_img,coeffs.detach().cpu(),render_result['lms_proj'].cpu().detach().numpy()


if __name__=='__main__':


    args = ImageFittingOptions()
    args = args.parse()

    device = 'cuda:0'


    mtcnn = MTCNN(select_largest=False, device=device)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)
    dst = '/home/allen/Documents/workplace/NeuralVoicePuppetry/media/frame_src_uvdata'


    #for index in range(len(os.listdir('/home/allen/Documents/workplace/NeuralVoicePuppetry/media/frame_src'))):
    for index in range(81,5864):

        img_name = f'{index:04d}'

        im = cv2.imread(f'/home/allen/Documents/workplace/NeuralVoicePuppetry/media/frames_jane/{img_name}.jpg')

        has_face, render_img ,resized_face_img ,coeffs,lms_proj = process_img(im, fa, mtcnn, device, args)

        im = Image.fromarray(render_img[0,...,:3].astype(np.uint8))

        img2 = Image.fromarray(resized_face_img)

        if has_face:        
            im.save(f"/home/allen/Documents/workplace/NeuralVoicePuppetry/media/framesJane_render/{img_name}_render.jpg")
            img2.save(f"/home/allen/Documents/workplace/NeuralVoicePuppetry/media/framesJane_render/{img_name}_crop.jpg")
            pickle.dump(coeffs, open(f'/home/allen/Documents/workplace/NeuralVoicePuppetry/media/framesJane_render/{img_name}_coeffs.pkl', 'wb'))
