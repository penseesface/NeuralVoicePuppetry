
import torch 

class Options():
    def __init__(self):
        self.rendererTypes = 'UNET_5_level' 
        # of gen filters in first conv layer
        self.ngf = 64
        self.norm = 'instance'
        self.no_dropout = True
        self.init_type='xavier'
        self.init_gain = 0.02
        self.gpu_ids = [0]

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) 

        self.model_path = '3d_fitting/inpainter/iter_15_inpainter_Jane.pth'
