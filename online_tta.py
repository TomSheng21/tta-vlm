import argparse

import time

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop, get_coop_dynamic
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, build_dataset_adv
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os

from online_method.clipzs import CLIPZS
from online_method.tda import TDA
from online_method.dmn import DMN
from online_method.dmn_weak import DMN_WEAK
from online_method.boostadapter import BoostAdapter
from online_method.dpe import DPE
from online_method.ecalp import ECALP
from online_method.onzeta import OnZeta
from online_method.dynaprompt import DynaPrompt

from PIL import ImageFilter, Image
import random

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def concat_dict(dict1, dict2):
    assert len(dict1.keys()) > 0
    for key in dict1.keys():
        if key in ['target']:
            dict1[key] = dict1[key].cpu()
        else:
            dict1[key] = dict1[key].cpu()
    if dict2 is None:
        return dict1
    else:
        assert dict2.keys() == dict1.keys()
        for key in dict1.keys():
            dict2[key] = torch.cat([dict2[key], dict1[key]], dim=0)
        return dict2
    
class MultiAugment:
    def __init__(self, strong_trans, weak_trans):
        self.strong = strong_trans
        self.weak = weak_trans
    
    def __call__(self, x):
        return self.strong(x), self.strong(x), self.weak(x) 

class GaussianBlur:
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def ColourDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def main():
    args = parser.parse_args()

    name_suffix = ''

    set_random_seed(args.seed)

    args.output_dir = os.path.join(args.output_dir, 'bs'+str(args.batch_size), args.arch, args.test_sets)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(os.path.join(args.output_dir, 'log'+name_suffix+'.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = None
    cudnn.benchmark = True

    # norm stats from clip.load() # NOTE: normalize are implemented in Model forward()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    dset = args.test_sets

    if True:
        if args.algorithm in ['tda', 'dmn_weak', 'ecalp', 'onzeta', 'clipzs']:
            args.data_aug = 'base'
            def _convert_image_to_rgb(image):
                return image.convert("RGB")
            data_transform = transforms.Compose([
                transforms.Resize(size=args.resolution, interpolation=BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(args.resolution, args.resolution)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size
            val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
            print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        elif args.algorithm in ['dmn', 'boostadapter', 'dpe', 'dynaprompt']:
            args.data_aug = 'single_multi_views'
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
                ])
            aug_bs = 64
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=aug_bs-1, 
                                            augmix=len(dset)>1)
            batchsize = args.batch_size
            val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
            print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        else:
            raise NotImplementedError
        
        args.loader_len = len(val_loader)

        print("evaluating: {}".format(dset))

        if len(dset) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(dset.lower()))
        else:
            assert dset in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if dset in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(dset.lower()))
                if dset == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        # ##########  Model  ##########

        if True:
            if args.algorithm in ['tda', 'dmn_weak', 'dmn', 'boostadapter', 'dpe', 'ecalp', 'onzeta', 'clipzs']:
                model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
            elif args.algorithm in ['dynaprompt']:
                num_p = 10
                model = get_coop_dynamic(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init, num_p=num_p)
            else:
                assert False
            if args.load is not None:
                print("Use pre-trained soft prompt (CoOp) as initialization")
                pretrained_ctx = torch.load(args.load, map_location='cpu')['state_dict']['ctx']
                assert pretrained_ctx.size()[0] == args.n_ctx
                if args.algorithm in ['dynaprompt']:
                    pretrained_ctx = pretrained_ctx.repeat(num_p, 1, 1)
                with torch.no_grad():
                    model.prompt_learner.ctx.copy_(pretrained_ctx)
                    model.prompt_learner.ctx_init_state = pretrained_ctx
            if args.load_tecoa:
                print('loading tecoa')
                args.robust_pretrain_path = {
                    'RN50': '/your_cache_path/tecoa/rn50_eps1.pth.tar',
                    'ViT-B/32': '/your_cache_path/tecoa/vitb32_eps4.pth.tar'
                }[args.arch]
                robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
                model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
            model_state = None

        print("=> Model created: visual backbone {}".format(args.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)


        if args.algorithm == 'tda':
            assert args.batch_size == 1
            tta_trainer = TDA(model, args.gpu)
        elif args.algorithm == 'dmn_weak':
            assert args.batch_size == 1
            tta_trainer = DMN_WEAK(model, args.gpu)
        elif args.algorithm == 'dmn':
            assert args.batch_size == 1
            tta_trainer = DMN(model, args.gpu)
        elif args.algorithm == 'boostadapter':
            tta_trainer = BoostAdapter(model, args.gpu)
        elif args.algorithm == 'dpe':
            tta_trainer = DPE(model, args.gpu)
        elif args.algorithm == 'ecalp':
            assert args.batch_size == 1
            tta_trainer = ECALP(model, args.gpu)
        elif args.algorithm == 'onzeta':
            tta_trainer = OnZeta(model, args.gpu)
        elif args.algorithm == 'dynaprompt':
            tta_trainer = DynaPrompt(model, args.gpu)
        elif args.algorithm == 'clipzs':
            assert args.batch_size == 1
            tta_trainer = CLIPZS(model, args.gpu)
        else:  
            raise NotImplementedError
        
        tta_trainer.prepare_model_and_optimization(args)

        save_dic = None

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        tta1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, tta1],
            prefix='Test: ')

        tta_trainer.model.eval()

        for i, (images, target) in enumerate(val_loader):
            end = time.time()
            assert args.gpu is not None

            if isinstance(images, list):
                if args.data_aug == 'batch_ssw_aug':
                    images = [img.cuda(args.gpu, non_blocking=True) for img in images]
                elif args.data_aug == 'single_multi_views':
                    for k in range(len(images)):
                        images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    image = images[0]
                    images = torch.cat(images, dim=0)
            else:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            tta_trainer.pre_adaptation()

            return_dict = tta_trainer.adaptation_process(None, images, args)

            # measure accuracy and record loss
            tpt_acc1, _ = accuracy(return_dict['output'], target, topk=(1, 5))

            return_dict['target'] = target

            if args.data_aug == 'batch_ssw_aug':
                tta1.update(tpt_acc1[0], images[0].size(0))
            else:
                tta1.update(tpt_acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            save_dic = concat_dict(return_dict, save_dic)

            if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
                print_log = 'iter:{}/{}, time={}, tta_acc1={}'.format(i, len(val_loader), batch_time.avg, tta1.avg)
                args.out_file.write(print_log + '\n')
                args.out_file.flush()
                print(print_log+'\n')
                progress.display(i)

            if (i+1) == len(val_loader):
                if len(name_suffix)>0:
                    torch.save(save_dic, os.path.join(args.output_dir, 'save_results_'+name_suffix+'.pt'))
                else:
                    torch.save(save_dic, os.path.join(args.output_dir, 'save_results.pt'))

        progress.display_summary()

        del val_dataset, val_loader
        print_log = "=> Acc. on testset [{}]: TTA Clean Acc @1 {}".format(dset, tta1.avg)
        save_log = {'tta_clean_acc': tta1.avg}
      
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

        torch.save(save_log, os.path.join(args.output_dir, 'results_log'+name_suffix+'.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', type=str, default='/data', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_results/temp')

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')

    parser.add_argument('--algorithm', type=str, default='clipzs', choices=['clipzs', 'tda', 'dmn_weak', 'dmn', 'boostadapter', 'dpe', 'ecalp', 'onzeta', 'dynaprompt'],)
    parser.add_argument('--load_tecoa', action='store_true')
    
    main()
