
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


        # self.model_path = 'checkpoints/10th_exp_trump_256_aug/35_inpainter.pth'
        # self.IMG_size = 256
        # self.ouput = 'media/4nd_english_audiov2_2_trump_256_aug_lmkavg'
        # self.target_path = 'media/frame_trump_fit_gt/train'
        # self.src_expression = 'media/expression_1230_trump_english.pkl'
        # self.background_v = 'media/trump_crop.mp4'
        # self.bbox = [460,11,805,356]#atrump
        # self.FPS = 30
        # self.mvg_lamda = 0.5
        # self.src_exp_lamda = 0.3



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
        
        
        
        # self.model_path = 'checkpoints/12th_exp_xiaoyan_256_aug/40_inpainter.pth'
        # self.IMG_size = 256
        # self.ouput = 'media/4nd_english_2_xiaoyan_256_audio_aug'
        # self.target_path = 'media/frame_xiaoyan_fit_gt/train'
        # self.src_expression = 'media/expression_1231_vid1_female_english.pkl'
        # self.background_v = 'media/tar_xiaoyan.MOV'
        # self.bbox = [250,629,859,1238]#aobama
        # self.FPS = 30
        # self.mvg_lamda = 0.5
        # self.src_exp_lamda = 0.3
        # self.cuthead = False
        
        
        
        # self.model_path = 'checkpoints/13th_exp_evelyn_256_aug/60_inpainter.pth'
        # self.IMG_size = 256
        # self.ouput = 'media/5th_english_2_evelyn_256_audio_aug_0118'
        # self.target_path = 'media/frame_evelyn_fit_gt/train'
        # self.src_expression = 'media/expression_0106_30fps.pkl'
        # self.background_v = 'media/src_evelyn.MOV'
        # self.bbox = [302,407,861,966]
        # self.FPS = 30
        # self.mvg_lamda = 0.3
        # self.src_exp_lamda = 0.5
        # self.cuthead = False   
        # self.trans_dth = 0.05
        # self.angle_dth = 0.015
        # self.smth_scale = 50
        
        
        
        # self.model_path = 'checkpoints/14th_exp_jane3_256_aug/40_inpainter.pth'
        # self.IMG_size = 256
        # self.ouput = 'media/4nd_english_2_jane3_256_audio_aug'
        # self.target_path = 'media/frame_jane3_fit_gt/train'
        # self.src_expression = 'media/expression_1231_vid1_female_english.pkl'
        # self.background_v = 'media/tar_jane_cut.mp4'
        # self.bbox = [261,509,503,751]#aobama
        # self.FPS = 30
        # self.mvg_lamda = 0.5
        # self.src_exp_lamda = 0.3
        # self.cuthead = False       
        
        
        
                
        # #single
        # self.model_path = 'checkpoints/19th_exp_audio_combine_256_aug2_fit/30_inpainter.pth'
        # self.IMG_size = 256
        # self.ouput = 'media/5th_0121_19audio_kpop'
        # self.target_path = 'media/frame_avata_farm_gt_fit_650/train'
        # self.src_expression = 'media/expression_0118_kpop.pkl'
        # self.background_v = 'media/src_farm_long.mp4'
        # self.bbox = [ 621,13,1314,706]
        # self.FPS = 30
        # self.mvg_lamda = 0.3
        # self.src_exp_lamda = 0.5
        # self.cuthead = False   
            
        # self.trans_dth = 0.05
        # self.angle_dth = 0.015
        # self.smth_scale = 50
        
        
        
        
        #double
        self.model_path1 = 'checkpoints/20th_exp_audio_double/15_inpainter.pth'
        self.model_path2 = 'checkpoints/20th_exp_audio_double/15_inpainter2.pth'
        self.IMG_size = 256
        self.ouput = 'media/6nd_0121double_19audio_kpop'
        self.target_path = 'media/frame_avata_farm_gt_fit_650/train'
        self.src_expression = 'media/expression_0118_kpop.pkl'
        self.background_v = 'media/src_farm_long.mp4'
        self.bbox = [ 621,13,1314,706]
        self.FPS = 30
        self.mvg_lamda = 0.3
        self.src_exp_lamda = 0.5
        self.cuthead = False   
            
        self.trans_dth = 0.05
        self.angle_dth = 0.015
        self.smth_scale = 50
        
        
        
        