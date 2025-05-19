import torch
from copy import deepcopy
import numpy as np
import operator
import math
from torch.nn import functional as F
from online_method.ecalp_utils import iterative_label_propagation



class ECALP():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    def prepare_model_and_optimization(self, args):
        self.model.eval()
        self.k_text = 3
        self.k_image = 8
        self.k_fewshot = 8
        self.gamma = 10.0
        self.alpha = 1.0
        self.num_iterations = 3
        self.image_features_cache = None

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features0, text_features, logit_scale = self.model.forward_features(images)
                assert image_features0.shape[0] == 1

                if self.image_features_cache is None:
                    self.image_features_cache = image_features0
                else:
                    self.image_features_cache = torch.cat((self.image_features_cache, image_features0), dim=0)

                image_features = torch.cat((self.image_features_cache, image_features0), dim=0)

                output = iterative_label_propagation(
                    image_features, text_features, self.k_text, self.k_image, self.k_fewshot,
                    self.gamma, self.alpha, fewshot_image_features=None,
                    fewshot_labels=None, max_iter=self.num_iterations)


        return_dict = {
            'output':output,
        }

        return return_dict



