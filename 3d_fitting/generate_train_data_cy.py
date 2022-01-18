import cv2
import numpy as np
import os
import pickle
from core import get_recon_model
from core.options import ImageFittingOptions
import torch
from PIL import Image
def resample(exp,src_rate,target_rate):
    L,D = exp.shape
    
    xp = np.arange(0,L/src_rate,1/src_rate).reshape(-1)
    
    x = np.arange(0,xp[-1],1/target_rate).reshape(-1)
    out = np.zeros([x.shape[0],D], dtype=np.float32)
    
    print(x.shape)
    print(xp.shape)
    print(exp[:,0].shape)
    
    if xp.shape[0] != exp[:,0].shape[0]:
        xp = xp[:-1]
        
    for i in range(D):
        buff = np.interp(x,xp,exp[:,i])
        out[:,i] = buff
    return out



def main():
    device = 'cuda:0'
    src_expression = pickle.load(open(f'media/expression_0118_farm_3min.pkl','br'))
    #src_expression = resample(src_expression,60,30)

    print(src_expression.shape)
    
    args = ImageFittingOptions()
    args = args.parse()
    recon_model = get_recon_model(model=args.recon_model, device=device, batch_size=1, img_size=args.tar_size)

    fit_folder = 'media/frame_avata_farm_gt_fit_650/train'
    
    dst = 'media/frame_avata_farm_gt_fit_cyaudio/train'
    
    for index,exp in enumerate(src_expression):
        
        coeffs = pickle.load(open(f'{fit_folder}/{index:04d}_coeffs.pkl','br'))        
        
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = recon_model.split_coeffs(coeffs)

        new_coeffes = recon_model.merge_coeffs( id_coeff.cuda(), torch.Tensor(exp).cuda().view(1,64), tex_coeff.cuda(), angles.cuda(), gamma.cuda(), translation.cuda() )
        
        render_result = recon_model(new_coeffes)
        
        render_img = render_result['rendered_img'].cpu().detach().numpy()
        
        crop_img = cv2.imread(f'{fit_folder}/{index:04d}_crop.jpg')
        crop_img = crop_img[:,:,::-1]
        lms_proj = render_result['lms_proj'].cpu().detach().numpy()
        
        
        im = Image.fromarray(render_img[0,...,:3].astype(np.uint8))

        img2 = Image.fromarray(crop_img)
        
        im.save(f"{dst}/{index:04d}_render.jpg")
        img2.save(f"{dst}/{index:04d}_crop.jpg")
        pickle.dump(coeffs, open(f'{dst}/{index:04d}_coeffs.pkl', 'wb'))
        pickle.dump(lms_proj, open(f'{dst}/{index:04d}_lms_proj.pkl', 'wb'))

    

if __name__  == '__main__':
    main()