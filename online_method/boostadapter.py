import torch
import operator
import math
from torch.nn import functional as F
import copy

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)
        if len(cache_keys) == 0:
            return torch.zeros(1)[0]

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().float()# .half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().float()# .half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # return alpha * cache_logits, affinity
        return alpha * cache_logits


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False, fifo=True):
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            cache[pred].append(item)
            if fifo: 
                if len(cache[pred])>shot_capacity:
                    cache[pred] = cache[pred][1:]
            else:
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
                cache[pred] = cache[pred][:shot_capacity]   
        else:
            cache[pred] = [item]

def select_confident_samples(feat, logits, topTPT):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    return feat[idxTPT], logits[idxTPT], idxTPT

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class BoostAdapter():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        self.pos_params = {
            'alpha': {'Caltech101':5.0,'DTD':2.0,'eurosat':4.0,'Food101':1.0,'Flower102':1.0,'Cars':1.0, 'UCF101':3.0, 'Pets':2.0, 'SUN397':2.0, 'Aircraft': 2.0, 'A':2.0, 'R':1.0, 'K':2.363, 'V':1.0, 'I':2.0, 'office-home':2.0, 'DomainNet':2.0},
            'beta':{'Caltech101':5.0,'DTD':3.0,'eurosat':8.0,'Food101':1.0,'Flower102':5.0,'Cars':7.0, 'UCF101':8.0, 'Pets':7.0, 'SUN397':3.0, 'Aircraft': 2.0, 'A':5.0, 'R':8.0, 'K':7.45, 'V':8.0, 'I':5.0, 'office-home':5.0, 'DomainNet':5.0}
        }
        self.delta_params = {
            'Caltech101': 2, 'DTD': 0, 'eurosat': 0, 'Food101': 3, 'Flower102': 0, 'Cars': 3,
            'UCF101': 0, 'Pets': 0, 'SUN397': 1, 'Aircraft': 2, 'A': 3, 'R': 3, 'K': 1,
            'V': 1, 'I': 1
        }

    def prepare_model_and_optimization(self, args):
        self.model.eval()

        self.use_pos_cache = True
        self.use_neg_cache = True

        self.pos_alpha = self.pos_params['alpha'][args.test_sets]
        self.pos_beta = self.pos_params['beta'][args.test_sets]
        self.delta = self.delta_params[args.test_sets]

        self.pos_shot_capacity = 3

        self.neg_alpha = 0.117
        self.neg_beta = 1.0

        self.pos_cache = {}
        self.neg_cache = {}

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features, text_features, logit_scale = self.model.forward_features(images)
                logits = logit_scale * image_features @ text_features.t()
                softmaxs = logits.softmax(dim=-1)
                softmax0 = softmaxs[0].unsqueeze(0)
                image_features0 = image_features[0].unsqueeze(0)
                ents = Entropy(softmaxs)
                ent0 = ents[0]
                pred0 = torch.max(logits[0].unsqueeze(0), 1)[1].item()
                num_classes = logits.size(1)

                select_feat, select_output, select_idx = select_confident_samples(image_features, logits, 0.1)
                select_entropy = ents[select_idx]


        update_cache(self.pos_cache, pred0, [image_features0, ent0], self.pos_shot_capacity, fifo=False)

        cur_pos_cache = copy.deepcopy(self.pos_cache)

        for i in range(select_entropy.shape[0]):
            cur_pred = int(select_output[i].argmax(dim=-1).item())
            cur_feat = select_feat[i]
            update_cache(cur_pos_cache, cur_pred, [cur_feat.unsqueeze(0), select_entropy[i].item()], self.pos_shot_capacity + self.delta, fifo=False)

        if 0.2 < ent0/math.log2(num_classes) and ent0/math.log2(num_classes) < 0.5:
            update_cache(self.neg_cache, pred0, [image_features0, ent0, softmax0], 2, True)

        output = logits[0].unsqueeze(0).clone()

        if self.use_pos_cache and len(self.pos_cache) > 0:
            output += compute_cache_logits(image_features0, cur_pos_cache, self.pos_alpha, self.pos_beta, text_features.unsqueeze(0))
        if self.use_neg_cache and len(self.neg_cache) > 0:
            output -= compute_cache_logits(image_features0, self.neg_cache, self.neg_alpha, self.neg_beta, text_features.unsqueeze(0), (0.03, 1.0))

        return_dict = {
            'output':output,
        }

        return return_dict



