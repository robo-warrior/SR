from argparse import ArgumentParser
from collections import defaultdict
from easydict import EasyDict as edict
from pprint import pprint
from prosr.data import DataLoader, Dataset # from prosr.data.progress_loader.py
from prosr.logger import info # from prosr.utils.logger.py
from prosr.models.trainer import CurriculumLearningTrainer, SimultaneousMultiscaleTrainer
from prosr.utils import get_filenames, IMG_EXTENSIONS, print_current_errors,set_seed ### Can't find ###
from time import time
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import skimage.io as io
from collections import OrderedDict
import numpy as np
import os
import os.path as osp
import prosr
import random
import sys
import torch
import yaml

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(osp.join(BASE_DIR, 'lib'))

# Plot PSNR

def parse_args():
    parser = ArgumentParser(description='training script for ProSR')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-m',
        '--model',
        type=str,
        help='model',
        choices=['prosr', 'prosrs', 'debug'])

    group.add_argument(
        '-c',
        '--config',
        type=str,
        help="Configuration file in 'yaml' format.")

    group.add_argument(
        '-ckpt',
        '--checkpoint',
        type=str,
        help='name of this training experiment',
    )
    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help="disable curriculum learning")

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='name of this training experiment',
        default=None)

    parser.add_argument(
        '--seed',
        type=int,
        help='reproducible experiments',
        default=128)

    parser.add_argument(
        '--fast-validation',
        type=int,
        help='truncate number of validation images',
        default=None)

    parser.add_argument(
        '-v',
        '--visdom',
        action='store_true',
        default=False)

    parser.add_argument(
        '-p',
        '--visdom-port',
        type=int,
        help='port used by visdom',
        default=8067)
    # 8067


    args = parser.parse_args()

    if (args.model or args.config) and args.output is None:
        parser.error("--model and --config requires --output.")

    ############# set up trainer ######################
    if args.checkpoint:
        args.output = osp.dirname(args.checkpoint)

    return args


def load_dataset(args):
    files = {'train':{},'test':{}}

    for phase in ['train','test']:
        for ft in ['source','target']:
            if args[phase].dataset.path[ft]:
                files[phase][ft] = get_filenames(
                    args[phase].dataset.path[ft], image_format=IMG_EXTENSIONS)
            else:
                files[phase][ft] = []

    return files['train'],files['test']


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def save_images_from_tensors(args, tensors, directory):
    n = tensors.shape[0]
    un_norm = UnNormalize(mean=args.train.dataset.mean,
                         std=args.train.dataset.stddev)
    for i in range(n):
        img_tensor = tensors[i]
        img_tensor = un_norm(img_tensor)
        torchvision.utils.save_image(img_tensor, directory + str(i + 1) + ".png")

def plot_psnr(args, psnr_list):
    # Plot psnr
    psnr_file = 'psnr_plot.png'
    plt.plot(psnr_list, label='PSNR')
    plt.xlabel('Epoch-1/' + str(args.train.io.eval_epoch_freq))
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig(psnr_file)
    plt.close()

def main(args):
    set_seed(args.cmd.seed)

    ############### loading datasets #################
    train_files,test_files = load_dataset(args)

    # reduce validation size for faster training cycles
    if args.test.fast_validation > -1:
        for ft in ['source','target']:
            test_files[ft] = test_files[ft][:args.test.fast_validation]

    info('training images = %d' % len(train_files['target']))
    info('validation images = %d' % len(test_files['target']))

    training_dataset = Dataset(
        prosr.Phase.TRAIN,
        **train_files,
        scale=args.data.scale,
        input_size=args.data.input_size,
        **args.train.dataset)

    training_data_loader = DataLoader(
        training_dataset, batch_size=args.train.batch_size)

    if len(test_files['target']):
        testing_dataset = Dataset(
                prosr.Phase.VAL,
                **test_files,
                scale=args.data.scale,
                input_size=None,
                **args.test.dataset)
        testing_data_loader = DataLoader(testing_dataset, batch_size=1)
    else:
        testing_dataset = None
        testing_data_loader = None


    if args.cmd.no_curriculum or len(args.data.scale) == 1:
        Trainer_cl = SimultaneousMultiscaleTrainer
    else:
        Trainer_cl = CurriculumLearningTrainer

    args.G.max_scale = np.max(args.data.scale)

    trainer = Trainer_cl(
        args,
        training_data_loader,
        save_dir=args.cmd.output,
        resume_from=args.cmd.checkpoint)

    log_file = os.path.join(args.cmd.output, 'loss_log.txt')

    steps_per_epoch = len(trainer.training_dataset)
    total_steps = trainer.start_epoch * steps_per_epoch


    ############# start training ###############
    info('start training from epoch %d, learning rate %e' %
         (trainer.start_epoch, trainer.lr))

    steps_per_epoch = len(trainer.training_dataset)
    errors_accum = defaultdict(list)
    errors_accum_prev = defaultdict(lambda: 0)

    # eval epochs incrementally
    eval_epoch_freq = 4
    batchsize = (int)(800/len(trainer.training_dataset))
    print("Batch size = ", batchsize)
    loss = []
    psnr_list = []
    output_imgs = torch.zeros((len(trainer.training_dataset)*batchsize, 3, 32, 32))

    #########################################################################
    for epoch in range(trainer.start_epoch + 1, args.train.epochs + 1):
        iter_start_time = time()
        epoch_start_time = time()
        trainer.set_train()
        epoch_loss = 0
        print("Epoch: ", epoch)
        # total_epoch_error = 0
        for i, data in enumerate(trainer.training_dataset):
            # data is a dictionary. See trainer.set_input() function for info
            # print("Batch", i)

            ##################################################################
            # Forward and backward pass
            trainer.set_input(**data)
            output_batch = trainer.forward()
            l1_loss = trainer.optimize_parameters()
            epoch_loss += l1_loss
            total_steps += 1

            ##################################################################
            # Save output images
            if(epoch % args.train.io.save_img_freq == 0 or epoch == args.train.epochs):
                # for ind in range(output_batch.shape[0]):
                #     output_np = trainer.tensor2imMine(output_batch[ind].detach())
                #     output_imgs[i*batchsize + ind] = output_np
                output_imgs[i*batchsize : (i+1)*batchsize] = output_batch

            ##################################################################
            # # Collect and print Errors (Unnecessary)
            # errors = trainer.get_current_errors()
            # for key, item in errors.items():
            #     errors_accum[key].append(item)
            #
            # if total_steps % args.train.io.print_errors_freq == 0:
            #     for key, item in errors.items():
            #         if len(errors_accum[key]):
            #             errors_accum[key] = np.nanmean(errors_accum[key])
            #         if np.isnan(errors_accum[key]):
            #             errors_accum[key] = errors_accum_prev[key]
            #     errors_accum_prev = errors_accum
            #     t = time() - iter_start_time
            #     iter_start_time = time()
            #     print_current_errors(
            #         epoch, total_steps, errors_accum, t, log_name=log_file)
            #
            #     if args.cmd.visdom:
            #         lrs = {
            #             'lr%d' % i: param_group['lr']
            #             for i, param_group in enumerate(
            #                 trainer.optimizer_G.param_groups)
            #         }
            #         real_epoch = float(total_steps) / steps_per_epoch
            #         visualizer.display_current_results(
            #             trainer.get_current_visuals(), real_epoch)
            #         visualizer.plot(errors_accum, real_epoch, 'loss')
            #         visualizer.plot(lrs, real_epoch, 'lr rate', 'lr')
            #
            #     errors_accum = defaultdict(list)

        ##################################################################
        # print loss and epoch time
        epoch_time = time() - epoch_start_time
        print("Epoch time = ", epoch_time)
        print("Epoch Loss per sample = ", epoch_loss/batchsize)
        loss.append(epoch_loss/batchsize)

        ##################################################################
        # Plot loss
        loss_file = 'l1_loss_plot.png'
        if(epoch%10 == 0):
            plt.plot(loss, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('L1 Loss')
            plt.legend()
            plt.savefig(loss_file)
            plt.close()

        ##################################################################
        # Save intermediate and final SR images
        # if(epoch % save_img_frequency == 0 or epoch == args.train.epochs):
        #     image_dir = "./outputs/output_images/"
        #     trainer.make_dir(image_dir)
        #     for i in range(output_imgs.shape[0]):
        #         image_name = '%d.png' % (i + 1)
        #         save_path = os.path.join(image_dir, image_name)
        #         image_pil = Image.fromarray(output_imgs[i].astype(np.uint8), mode='RGB')
        #         image_pil.save(save_path)
                # io.imsave(save_path, output_img)

        if(epoch % args.train.io.save_img_freq == 0 or epoch == args.train.epochs):
            image_dir = "./outputs/output_images/"
            trainer.make_dir(image_dir)
            save_images_from_tensors(args, output_imgs, image_dir)

        ##################################################################
        # Save model
        if epoch % args.train.io.save_model_freq == 0:
            info(
                'saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps),
                bold=True)
            trainer.save(str(epoch), epoch, trainer.lr)

        ##################################################################
        # update learning rate
        if (epoch - trainer.best_epoch) > args.train.lr_schedule_patience:
            trainer.save('last_lr_%g' % trainer.lr, epoch, trainer.lr)
            trainer.update_learning_rate()

        ##################################################################
        # test with validation set, PSNR calculation
        if testing_data_loader and (((epoch) % args.train.io.eval_epoch_freq==0 or epoch==args.train.epochs) or epoch==1):
            # eval_epoch_freq = min(eval_epoch_freq * 2, args.train.io.eval_epoch_freq)
            with torch.no_grad():
                test_start_time = time()
                # use validation set
                trainer.set_eval()
                trainer.reset_eval_result()
                for i, data in enumerate(testing_data_loader):
                    trainer.set_input(**data)
                    trainer.evaluate()

                t = time() - test_start_time
                test_result = trainer.get_current_eval_result()

                ################ visualize ###############
                if args.cmd.visdom:
                    visualizer.plot(test_result,
                                    float(total_steps) / steps_per_epoch,
                                    'eval', 'psnr')

                trainer.update_best_eval_result(epoch, test_result)
                info(
                    'eval at epoch %d : ' % epoch + ' | '.join([
                        '{}: {:.02f}'.format(k, v)
                        for k, v in test_result.items()
                    ]) + ' | time {:d} sec'.format(int(t)),
                    bold=True)
                for k, v in test_result.items():
                    psnr_list.append(v)

                info(
                    'best so far %d : ' % trainer.best_epoch + ' | '.join([
                        '{}: {:.02f}'.format(k, v)
                        for k, v in trainer.best_eval.items()
                    ]),
                    bold=True)

                if trainer.best_epoch == epoch:
                    if len(trainer.best_eval) > 1:
                        if not isinstance(trainer, CurriculumLearningTrainer):
                            best_key = [
                                k for k in trainer.best_eval
                                if trainer.best_eval[k] == test_result[k]
                            ]
                        else:
                            # select only upto current training scale
                            best_key = ["psnr_x%d" % trainer.opt.data.scale[s_idx]
                                    for s_idx in range(trainer.current_scale_idx+1)]
                            best_key = [k for k in best_key
                                    if trainer.best_eval[k] == test_result[k]]

                    else:
                        best_key = list(trainer.best_eval.keys())
                    trainer.save(str(epoch) + '_best_' + '_'.join(best_key), epoch,
                                 trainer.lr)

            plot_psnr(args, psnr_list)

    ##################################################################
    # Plot final loss
    loss_file = 'l1_loss_plot.png'
    plt.plot(loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.savefig(loss_file)
    plt.close()
    ##################################################################



if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    if args.config is not None:
        with open(args.config) as stream:
            try:
                params = edict(yaml.load(stream))
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(0)
    elif args.model is not None:
        params = edict(getattr(prosr, args.model + '_params'))

    else:
        # params = torch.load(args.checkpoint + '_net_G.pth')['params']
        params = torch.load(args.checkpoint)['params']

    # parameters overring
    if args.fast_validation is not None:
        params.test.fast_validation = args.fast_validation
    del args.fast_validation

    # Add command line arguments
    params.cmd = edict(vars(args))

    pprint(params)

    if not osp.isdir(args.output):
        os.makedirs(args.output)
    np.save(osp.join(args.output, 'params'), params)

    experiment_id = osp.basename(args.output)

    info('experiment ID: {}'.format(experiment_id))

    if args.visdom:
        from prosr.visualizer import Visualizer
        visualizer = Visualizer(experiment_id, port=args.visdom_port)

    main(params)