
import torch 

class Options():
    def __init__(self):
        self.rendererTypes = 'UNET_8_level' 
        # of gen filters in first conv layer
        self.ngf = 64
        self.norm = 'instance'
        self.no_dropout = True
        self.init_type='xavier'
        self.init_gain = 0.02
        self.gpu_ids = [0]

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) 

        self.model_path = 'checkpoints/10th_exp_trump_256_aug/35_inpainter.pth'
        self.IMG_size = 256
        self.ouput = 'media/4nd_chinese_fit_2_trump_256_aug_lmkavg'
        self.target_path = 'media/frame_trump_fit_gt/train'
        self.src_expression = 'media/frame_src_trump_chinese_fit_gt/train'
        self.background_v = 'media/trump_crop.mp4'
        self.bbox = [460,11,805,356]#atrump
        self.FPS = 30

        self.mvg_lamda = 0.5
        self.src_exp_lamda = 0.3

        # self.model_path = 'checkpoints/11th_exp_obama_256_aug/40_inpainter.pth'
        # self.IMG_size = 256
        # self.ouput = 'media/4nd_english_2_obama_256_audio_aug'

        # self.target_path = 'media/frames_obama4_fit_gt2/train'

        # self.src_expression = 'media/expression_1221.pkl'

        # self.background_v = 'media/obama3.mp4'

        # self.bbox = [382,13,831,462]#aobama
        # self.FPS = 30
        # self.mvg_lamda = 0.5
        # self.src_exp_lamda = 0.3