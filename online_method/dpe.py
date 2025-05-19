import torch
from copy import deepcopy
import numpy as np
import operator
import math
from torch.nn import functional as F
import copy
from info_nce import InfoNCE

def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
        return
    
def cache_key_value(image_features, cache, alpha, beta, clip_weights):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            # Compute the prototype of the class
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)


        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
            
        return cache_keys, cache_values, all_classes
    
def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits
    
class TextResidue(torch.nn.Module):
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.residual = torch.nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights
    
    def reset(self):
        self.residual = torch.nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
class PositiveCacheResidue(torch.nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = torch.nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys

class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def select_confident_samples(feat, logits, topTPT):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    # return feat[idxTPT], logits[idxTPT]
    return feat[idxTPT], logits[idxTPT], idxTPT

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class DPE():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        self.pos_params = {
            'alpha': {'Caltech101':15.0,'DTD':6.0,'eurosat':3.0, 'Aircraft':6.0, 'Food101':3.0, 'Flower102':2.5,'Cars':3.0, 'UCF101':9.0, 'Pets':4.0, 'SUN397':6.0,  'A':6.0, 'R':3.0, 'K':7.0, 'V':3.0, 'I':6.0},
            'beta':{'Caltech101':5.0,'DTD':3.0,'eurosat':8.0, 'Aircraft':2.0, 'Food101':1.0, 'Flower102':5.0,'Cars':7.0, 'UCF101':8.0, 'Pets':7.0, 'SUN397':3.0, 'A':5.0, 'R':8.0, 'K':7.45, 'V':8.0, 'I':5.0}
        }
        self.lr_params = {
            'text':{'Caltech101':0.0006,'DTD':0.0006, 'eurosat':0.00005, 'Aircraft':0.0006, 'Food101':0.0002, 'Flower102':0.0001, 'Cars':0.0001, 'UCF101':0.0004, 'Pets':0.00005, 'SUN397':0.0002, 'A':0.0003, 'R':0.0006, 'K':0.0006, 'V':0.0005, 'I':0.0006},
            'image':{'Caltech101':0.0006,'DTD':0.0006, 'eurosat':0.00005, 'Aircraft':0.0006, 'Food101':0.0002, 'Flower102':0.0001, 'Cars':0.0001, 'UCF101':0.0004, 'Pets':0.00005, 'SUN397':0.0002, 'A':0.0003, 'R':0.0006, 'K':0.0006, 'V':0.0005, 'I':0.0006},
            'align':{'Caltech101':0.2,'DTD':0.2, 'eurosat':0.2, 'Aircraft':0.2, 'Food101':0.2, 'Flower102':0.2, 'Cars':1.5, 'UCF101':0.2, 'Pets':0.2, 'SUN397':0.0002, 'A':2.5, 'R':0.0, 'K':0.2, 'V':0.5, 'I':0.5}
        }

    def prepare_model_and_optimization(self, args):
        self.model.eval()

        self.use_pos_cache = True
        self.use_neg_cache = True

        self.pos_alpha = self.pos_params['alpha'][args.test_sets]
        self.pos_beta = self.pos_params['beta'][args.test_sets]
        self.text_lr = self.lr_params['text'][args.test_sets]
        self.image_lr = self.lr_params['image'][args.test_sets]
        self.align = self.lr_params['align'][args.test_sets]

        self.pos_shot_capacity = 3

        self.pos_cache = {}

        # self.clip_weights = self.model.get_text_features().t().clone()
        self.clip_weights_global = self.model.get_text_features().t().clone()
        self.num_avg = 0
 

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):
        clip_weights_local = self.clip_weights_global.clone().detach()
        text_residue = TextResidue(clip_weights_local)
        new_clip_weights = text_residue(clip_weights_local)
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # image_features, text_features, logit_scale = self.model.forward_features(images)
                # image_features = self.model.image_encoder(images.type(self.model.dtype)) # maple bug
                image_features, _, _ = self.model.forward_features(images.type(self.model.dtype))
                logit_scale = self.model.logit_scale.exp()

                logits = logit_scale * image_features @ new_clip_weights
                _, _, select_idx = select_confident_samples(image_features, logits, 0.1)
                image_features_x = image_features[select_idx].mean(0).unsqueeze(0)
                clip_logits = logits[select_idx].mean(dim=0).unsqueeze(0)
                prob_map = logits[select_idx].softmax(1).mean(0).unsqueeze(0)
                pred = int(logits[select_idx].mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
                entropy = Entropy(clip_logits.softmax(dim=-1))

            num_classes = logits.size(1)
            entropy = entropy/math.log2(num_classes) 
            update_cache(self.pos_cache, pred, [image_features_x, entropy], self.pos_shot_capacity)
            pos_cache_keys, pos_cache_values, all_classes = cache_key_value(image_features_x, self.pos_cache, self.pos_alpha, self.pos_beta, new_clip_weights)
            pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
            steps = 1

            for j in range(steps):

                optimizer = torch.optim.AdamW([
                    {'params': text_residue.parameters(), 'lr': self.text_lr, 'eps': 1e-3, 'weight_decay': 1e-1},
                    {'params': pos_cache_residue.parameters(), 'lr': self.image_lr, 'eps': 1e-3, 'weight_decay': 1e-1}
                ])

                optimizer.zero_grad()


                new_clip_weights = text_residue(clip_weights_local)
                final_logits = clip_logits.clone()

                new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                final_logits += compute_cache_logits(image_features_x, new_pos_cache_keys, pos_cache_values, self.pos_alpha, self.pos_beta, new_clip_weights)
                loss = avg_entropy(final_logits)
                # alignment loss
                image2text_loss = InfoNCELoss(new_pos_cache_keys.T, new_clip_weights[:, all_classes].T)
                loss += image2text_loss * self.align

                loss.backward()
                optimizer.step()

                # self.scaler.scale(loss).backward()
                # self.scaler.step(optimizer)
                # self.scaler.update()


            text_residue.eval()
            pos_cache_residue.eval()
            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                if args.test_sets == 'A':
                    # image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, new_clip_weights)
                    
                    clip_logits = 100. * image_features @ new_clip_weights
                    _, _, select_idx = select_confident_samples(image_features, clip_logits, 0.1)
                    clip_logits = clip_logits[select_idx].mean(0).unsqueeze(0)

                    image_features = image_features_x.mean(0).unsqueeze(0)
                else:
                    # image_features, clip_logits, _, _, _ = get_clip_logits(images[0], clip_model, new_clip_weights)
                    image_features = image_features[0].unsqueeze(0)
                    clip_logits = 100. * image_features @ new_clip_weights
                final_logits = clip_logits.clone()
                new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                final_logits += compute_cache_logits(image_features, new_pos_cache_keys, pos_cache_values, self.pos_alpha, self.pos_beta, new_clip_weights)       
            
                loss = avg_entropy(final_logits)

                if loss/math.log2(num_classes)  < 0.1:
                    self.num_avg += 1
                    self.clip_weights_global = self.clip_weights_global * (self.num_avg / (self.num_avg + 1)) + new_clip_weights * (1 / (self.num_avg + 1))

        return_dict = {
            'output':final_logits,
        }

        return return_dict
