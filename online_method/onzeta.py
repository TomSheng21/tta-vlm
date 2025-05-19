import torch
from copy import deepcopy
import numpy as np
import operator
import math
from torch.nn import functional as F
import copy


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class OnZeta():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)
        
        

    def prepare_model_and_optimization(self, args):
        self.model.eval()

        self.cw = 0.5
        self.cr = 20
        self.beta = 0.8
        self.alpha = 1
        self.tau_i = 0.04
        self.tau_t = 0.01

        if len(args.test_sets) <= 1:
            self.beta = 0.8
        else:
            self.beta = 0.4

        if args.test_sets in ['Caltech101']:
            self.alpha = 0

        self.clip_weights = self.model.get_text_features().t().clone()
        self.num_class = self.clip_weights.shape[1]

        self.w = self.model.get_text_features().t().clone()
        self.rho = torch.zeros(self.num_class).unsqueeze(0).cuda(self.device)
        
        self.ii = 0
        

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):       
        lr = self.cw / math.sqrt(self.ii + 1)
        rlr = self.cr / math.sqrt(self.ii + 1)
        beta = self.beta * math.sqrt((self.ii + 1) / args.loader_len)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # image_features, text_features, logit_scale = self.model.forward_features(images)
                # image_features = self.model.image_encoder(images.type(self.model.dtype))
                image_features, _, _ = self.model.forward_features(images.type(self.model.dtype))
                logit_scale = self.model.logit_scale.exp()
                tlabel = F.softmax(image_features @ self.clip_weights / self.tau_t, dim=-1)
                tlabel = tlabel * torch.exp(self.rho)
                tlabel /= torch.sum(tlabel)
                self.rho -= rlr * (tlabel - self.alpha / self.num_class)
                self.rho[self.rho < 0] = 0

                vision_label = F.softmax(image_features @ self.w / self.tau_i, dim=-1)
                output = beta * vision_label + (1 - beta) * tlabel
                grad = torch.outer(image_features.squeeze(), (vision_label - tlabel).squeeze())
                self.w -= (lr / self.tau_i) * grad
                self.w = F.normalize(self.w, dim=0)

        self.ii += 1    

        return_dict = {
            'output':output,
        }

        return return_dict
