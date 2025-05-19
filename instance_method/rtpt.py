import torch
from copy import deepcopy
import numpy as np

def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def get_top_sim(sim_matrix):
    k = 20
    sim_matrix[sim_matrix>=1.0] = float('-inf')
    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    top_k_mean = top_k_values.mean(dim=-1)
    return top_k_mean

class RTPT():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    def prepare_model_and_optimization(self, args):
        self.model.eval()

        trainable_param = self.model.prompt_learner.parameters()
        self.optimizer = torch.optim.AdamW(trainable_param, args.lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())

    def pre_adaptation(self):
        with torch.no_grad():
            self.model.reset()
        self.optimizer.load_state_dict(self.optim_state)
    
    def adaptation_process(self, image, images, args):
        assert args.tta_steps > 0

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                clip_features, _, _ = self.model.forward_features(images)

        selected_idx = None
        for j in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                output = self.model(images) 

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, args.selection_p)

                loss = entropy_avg(output)

                self.optimizer.zero_grad()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                tuned_outputs = self.model(images)
                tta_prompt_embedding = self.model.prompt_learner.ctx.detach()

                sim_matrix_images = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
                score = get_top_sim(sim_matrix_images)
                weight = torch.nn.functional.softmax(score/0.01, dim=-1)
                tta_output = torch.bmm(weight.unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)

        return_dict = {
            'output':tta_output,
            'tta_prompt_embedding':tta_prompt_embedding,
        }

        return return_dict



