import time
import copy
import torch
from options.train_options import TrainOptions
from options.test_options import TestOptions
from models import create_model
from util.visualizer import Visualizer
from data import CreateDataLoader
from PIL import Image
import numpy as np
import cv2
from util import util

if __name__ == '__main__':
    # training dataset
    opt = TestOptions().parse()

    # model
    model = create_model(opt)
    model.setup(opt)
    

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)


    if opt.renderer != 'no_renderer':
        print('load renderer')
        model.loadModules(opt, opt.renderer, ['netD','netG'])


    validation_error = 0
    cnt = 0
    for i, data in enumerate(dataset):
        print(data['ID'])
        model.set_input(data)
        model.forward()

        result =util.tensor2im( model.fake)

        num = data['ID'].cpu().detach()

        # cv2.imshow('',result)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        Image.fromarray(result).save(f'media/transfer_render/{num[0]:04d}_render.jpg')
