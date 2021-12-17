
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

        self.model_path = 'checkpoints/7th_exp_obama_512/latest_inpainter.pth'

        self.IMG_size = 512

        self.ouput = 'media/3nd_version_virtual_zhubo_2_obama_512_v2'

        self.target_path = 'media/frames_obama3_fit_gt/train'

        self.src_expression = 'media/frames_zhubo_fit_gt'

        self.background_v = 'media/obama3.mp4'