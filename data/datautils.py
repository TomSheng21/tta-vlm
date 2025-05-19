import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'imagenet/images',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'imagenet-sketch/images',
    'R': 'imagenet-rendition/imagenet-r',
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'flower102': 'few-shot-datasets/oxford_flowers',
    'dtd': 'few-shot-datasets/dtd',
    'pets': 'few-shot-datasets/oxford_pets',
    'cars': 'few-shot-datasets/stanford_cars',
    'ucf101': 'few-shot-datasets/ucf101',
    'caltech101': 'few-shot-datasets/caltech-101',
    'food101': 'few-shot-datasets/food-101',
    'sun397': 'few-shot-datasets/sun397',
    'aircraft': 'few-shot-datasets/fgvc_aircraft',
    'eurosat': 'few-shot-datasets/eurosat'
}

class ImageFolder_path(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform,
    ):
        super().__init__(
            root=root,
            transform=transform
        )
        self.imgs = self.samples
    

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, torch.tensor(target).long(), path

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset

# def __init__(
#         self,
#         root: Union[str, Path],
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         loader: Callable[[str], Any] = default_loader,
#         is_valid_file: Optional[Callable[[str], bool]] = None,
#         allow_empty: bool = False,
#     ):
#         super().__init__(
#             root,
#             loader,
#             IMG_EXTENSIONS if is_valid_file is None else None,
#             transform=transform,
#             target_transform=target_transform,
#             is_valid_file=is_valid_file,
#             allow_empty=allow_empty,
#         )
#         self.imgs = self.samples


# class ImageFolder_adv(datasets.ImageFolder):
#     def __init__(
#             self,
#             root=None,
#             transform=None,
#             replace_path=None,
#     ):
#         self.replace_path = replace_path
#         super().__init__(
#             root=root,
#             transform=transform,
#         )

#     def __getitem__(self, index: int):
#             """
#             Args:
#                 index (int): Index

#             Returns:
#                 tuple: (sample, target) where target is class_index of the target class.
#             """
#             path, target = self.samples[index]
#             # 'jpg' or name_list[-1] == 'JPEG' or name_list[-1] == 'jpeg'

#             path = path.replace('.jpg','.png').replace('JPEG', '.png').replace('.jpeg','.png')
#             path = path.replace('/data/shenglijun/dataset/', '/data/shenglijun/dataset/'+self.replace_path)
#             sample = self.loader(path)
#             if self.transform is not None:
#                 sample = self.transform(sample)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)

#             return sample, target

def build_dataset_adv(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, replace_path=''):
    assert len(replace_path) > 0
    if set_id in fewshot_datasets or set_id == 'I':
        if mode == 'train' and n_shot:
            assert False
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset_adv(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, replace_path=replace_path)
    # elif set_id == 'I':
    #     # ImageNet validation set
    #     testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
    #     testdir = testdir.replace('/data/shenglijun/dataset/', '/data/shenglijun/dataset/'+replace_path+'/')
    #     testset = datasets.ImageFolder(testdir, transform=transform)
    # elif set_id in ['A', 'K', 'R', 'V']:
    #     testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
    #     testdir = testdir.replace('/data/shenglijun/dataset/', '/data/shenglijun/dataset/'+replace_path+'/')
    #     testset = datasets.ImageFolder(testdir, transform=transform)
    else:
        raise NotImplementedError
    return testset

def build_dataset_path(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    # for adv generation
    if set_id in fewshot_datasets:
        testset = build_fewshot_dataset_path(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'I':
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        print(testdir)
        testset = ImageFolder_path(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        print(testdir)
        testset = ImageFolder_path(testdir, transform=transform)
    else:
        raise NotImplementedError
    return testset

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views

class MultiAugmenter(object):
    def __init__(self, base_transform, n_views_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views_transform = n_views_transform
        self.n_views = n_views

        
    def __call__(self, x):
        image = self.base_transform(x)
        views = [self.n_views_transform(x) for _ in range(self.n_views)]
        return [image] + views


class Post_AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views


class AugMixAugmenter_SigLip(object):
    def __init__(self, preprocessor, n_views=2, augmix=False, severity=1):
        self.preprocessor = preprocessor
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocessor(images=(x), return_tensors="pt", padding=True).pixel_values[0]
        views = [self.preprocessor(images=augmix_siglip(x, None, self.aug_list, self.severity), return_tensors="pt", padding=True).pixel_values[0] for _ in range(self.n_views)]
        return [image] + views


def augmix_siglip(image, preprocessor, aug_list, severity=1):
    
    preaugment = get_preaugment()
    x_orig = (preaugment(image))
    if len(aug_list) == 0:
        return x_orig
    x_processed = np.array(x_orig)
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = np.zeros_like((x_processed))
    blended = np.zeros_like(x_processed, dtype=np.float32)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        blended += np.array(x_aug).astype(np.float32) * w[i]
    
    mix = m * x_processed + (1 - m) * blended
    mix_img = Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8))
    # mix_img.save("blended.jpg")
    return mix_img


