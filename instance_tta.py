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

from clip.custom_clip import get_coop, get_shift_v2, get_clip_lora_v2
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, build_dataset_adv
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os

from instance_method.tpt import TPT
from instance_method.ctpt import CTPT
from instance_method.mta import MTA
from instance_method.rlcf import RLCF
from instance_method.clipzs import CLIPZS
from instance_method.zero import ZERO
from instance_method.tps import TPS
from instance_method.ttl import TTL
from instance_method.rtpt import RTPT

import time

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
            dict1[key] = dict1[key].unsqueeze(0).cpu()
    if dict2 is None:
        return dict1
    else:
        assert dict2.keys() == dict1.keys()
        for key in dict1.keys():
            dict2[key] = torch.cat([dict2[key], dict1[key]], dim=0)
        return dict2

def main():
    args = parser.parse_args()

    name_suffix = ''

    set_random_seed(args.seed)

    args.output_dir = os.path.join(args.output_dir, args.arch, 'seed_'+str(args.seed), args.test_sets)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(os.path.join(args.output_dir, 'log'+name_suffix+'.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    cudnn.benchmark = True

    # norm stats from clip.load() # NOTE: normalize are implemented in Model forward()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    dset = args.test_sets

    if True:
        if args.algorithm in ['tpt', 'mta', 'ctpt', 'rlcf', 'tps', 'ttl', 'rtpt', 'zero', 'clipzs']:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
                ])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(dset)>1)
            batchsize = 1
            val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
            print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        else:
            raise NotImplementedError

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
            if args.algorithm in ['tpt', 'mta', 'ctpt', 'rlcf', 'clipzs', 'zero', 'rtpt']:
                model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
            elif args.algorithm in ['tps']:
                model = get_shift_v2(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
            elif args.algorithm in ['ttl']:
                model = get_clip_lora_v2(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
            if args.load is not None:
                print("Use pre-trained soft prompt (CoOp) as initialization")
                pretrained_ctx = torch.load(args.load, map_location='cpu')['state_dict']['ctx']
                assert pretrained_ctx.size()[0] == args.n_ctx
                with torch.no_grad():
                    model.prompt_learner.ctx.copy_(pretrained_ctx)
                    model.prompt_learner.ctx_init_state = pretrained_ctx
            if args.load_tecoa:
                print('loading tecoa')
                args.robust_pretrain_path = {
                    'RN50': 'your_cache_path/tecoa/rn50_eps1.pth.tar',
                    'ViT-B/32': 'your_cache_path/tecoa/vitb32_eps4.pth.tar'
                }[args.arch]
                robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
                model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])

        if args.algorithm in ['tps']:
            model.add_shifter()
        if args.algorithm in ['ttl']:
            model.add_lora()

        print("=> Model created: visual backbone {}".format(args.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

        if args.algorithm == 'tpt':
            tta_trainer = TPT(model, args.gpu)
        elif args.algorithm == 'ctpt':
            tta_trainer = CTPT(model, args.gpu)
        elif args.algorithm == 'mta':
            tta_trainer = MTA(model, args.gpu)
        elif args.algorithm == 'rlcf':
            tta_trainer = RLCF(model, args.gpu)
        elif args.algorithm == 'clipzs':
            tta_trainer = CLIPZS(model, args.gpu)
        elif args.algorithm == 'zero':
            tta_trainer = ZERO(model, args.gpu)
        elif args.algorithm == 'tps':
            tta_trainer = TPS(model, args.gpu)
        elif args.algorithm == 'ttl':
            tta_trainer = TTL(model, args.gpu)
        elif args.algorithm == 'rtpt':
            tta_trainer = RTPT(model, args.gpu)
        else:  
            raise NotImplementedError
        
        tta_trainer.prepare_model_and_optimization(args)

        save_dic = None

        data_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        reset_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        adapt_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        tta1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [data_time, reset_time, adapt_time, top1, tta1],
            prefix='Test: ')

        tta_trainer.model.eval()

        end = time.time()

        for i, (images, target) in enumerate(val_loader):

            assert args.gpu is not None
            target = target.cuda(args.gpu, non_blocking=True)

            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(args.gpu, non_blocking=True)
                image = images
            
            images = torch.cat(images, dim=0)

            data_end_time = time.time()

            tta_trainer.pre_adaptation()

            reset_end_time = time.time()

            with torch.no_grad():
                clip_output = tta_trainer.model(image)
                clip_features, _, _ = tta_trainer.model.forward_features(images)
                clip_outputs = tta_trainer.model(images)

            return_dict = tta_trainer.adaptation_process(image, images, args)

            acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))
            tpt_acc1, _ = accuracy(return_dict['output'], target, topk=(1, 5))

            return_dict['clip_output'] = clip_output
            return_dict['clip_features'] = clip_features
            return_dict['clip_outputs'] = clip_outputs
            return_dict['target'] = target
        
            top1.update(acc1[0], images.size(0))
            tta1.update(tpt_acc1[0], images.size(0))

            adapt_end_time = time.time()
            
            data_time.update(data_end_time-end)
            reset_time.update(reset_end_time-data_end_time)
            adapt_time.update(adapt_end_time-reset_end_time)
            

            save_dic = concat_dict(return_dict, save_dic)

            if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
                print_log = 'iter:{}/{}, time={}/{}/{}, clip_acc1={}, tta_acc1={}'.format(i, len(val_loader), data_time.avg, reset_time.avg, adapt_time.avg, top1.avg, tta1.avg)
                args.out_file.write(print_log + '\n')
                args.out_file.flush()
                print(print_log+'\n')
                progress.display(i)

            if (i+1) == len(val_loader):
                torch.save(save_dic, os.path.join(args.output_dir, 'save_results.pt'))

            end = time.time()

            

        progress.display_summary()

        del val_dataset, val_loader
        print_log = "=> Acc. on testset [{}]: Clean Acc @1 {}/ TTA Clean Acc @1 {}".format(dset, top1.avg, tta1.avg)
        save_log = {'clean_acc': top1.avg, 'tta_clean_acc': tta1.avg}
      
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
    parser.add_argument('-p', '--print-freq', default=200, type=int,
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

    parser.add_argument('--algorithm', type=str, default='tpt', choices=['tpt', 'mta', 'ctpt', 'rlcf', 'clipzs', 'zero', 'tps', 'ttl', 'rtpt'])

    parser.add_argument('--load_tecoa', action='store_true')

    main()
