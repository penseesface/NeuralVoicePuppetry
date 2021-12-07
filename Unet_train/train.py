import time
import copy
import torch
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from data import CreateDataLoader

if __name__ == '__main__':
    # training dataset
    opt = TrainOptions().parse()

    # model
    model = create_model(opt)
    model.setup(opt)

    print(model)
    

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)


    if opt.renderer != 'no_renderer':
        print('load renderer')
        model.loadModules(opt, opt.renderer, ['netD','netG'])

    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0 # iterator within an epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_input(data)
            model.optimize_parameters(epoch)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()


                # if opt.compute_val: 
                #     validation_error = 0
                #     cnt = 0
                #     for i, data in enumerate(dataset_validation):
                #         model.set_input(data)
                #         model.forward()
                #         model.backward_G(epoch) # be carefull with the gradients (are zeroed in the optimization step)
                #         validation_error += model.loss_G.detach().cpu()
                #         cnt += 1.0
                #     validation_error /= cnt
                #     #print('Validation Error:', validation_error)
                #     #visualizer.plot_current_validation_error(epoch, float(epoch_iter) / dataset_size, {'validation_error': validation_error})
                #     losses.update({'validation_error': validation_error})

                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()



        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
