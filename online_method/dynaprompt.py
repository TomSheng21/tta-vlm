import torch
from copy import deepcopy
import numpy as np


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_confident_samples(logits, top):
    # pdb.set_trace()
    batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    if batch_entropy.dim() > 1:
        batch_entropy = batch_entropy.mean(-1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(logits.size()[0] * top)]
    return logits[idx], idx

def entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * logits.exp()).sum(dim=-1)

class DynaPrompt():
    def __init__(self, model, device, num_p):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)
        self.num_prompts = num_p # also in get_coop_dynamic
        self.aug_size = 64

    def prepare_model_and_optimization(self, args):
        self.model.eval()

        if len(args.test_sets)<=1:
            lr = 5e-3
        else:
            lr = 3e-3

        trainable_param = self.model.prompt_learner.parameters()
        self.optimizer = torch.optim.AdamW(trainable_param, lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):
        assert args.tta_steps > 0

        image = images[0].unsqueeze(0)

        with torch.no_grad():
            if self.model.prompt_learner.ctx_use.count(0) == 0:
                self.model.prompt_learner.ctx_use[self.model.prompt_learner.ctx_order[0]] = 0
                self.model.prompt_learner.ctx[self.model.prompt_learner.ctx_order[0]] = self.model.prompt_learner.ctx[self.model.prompt_learner.ctx_order[0]] * 0

        for j in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                output = self.model(images) 
                p_pred = output.view(output.size()[0], self.num_prompts, -1).detach()

                p_ent = entropy(p_pred).mean(0)
                plpd = p_pred[0].max(-1)[0].unsqueeze(0) - p_pred[1:].max(dim=-1)[0]
                plpd = plpd.mean(0)

                ent_order = p_ent.topk(self.num_prompts)[1]
                init_p_position = self.model.prompt_learner.ctx_order[0]
                exactnumber = torch.where(ent_order==init_p_position)[0].item()

                plpd_order = plpd.topk(self.num_prompts)[1]
                exactnumberlist1 = ent_order[min(exactnumber + 1, self.num_prompts - 1):]
                exactnumberlist2 = plpd_order[:exactnumber]
                alllist = np.intersect1d(exactnumberlist1.cpu().numpy(), exactnumberlist2.cpu().numpy())

                if set(alllist):
                    exactnumber = torch.cat([ent_order[alllist], ent_order[[exactnumber]]], dim=0).cpu().numpy() # with init prompt
                else:
                    exactnumber = ent_order[exactnumber].cpu().numpy()

                output = output.view(self.aug_size, self.num_prompts, -1)[:, exactnumber]
                exactnumber_list = np.array(exactnumber).reshape(-1).tolist()
                for i in exactnumber_list:
                    self.model.prompt_learner.ctx_order.remove(i)
                    self.model.prompt_learner.ctx_use[i] += 1
                    self.model.prompt_learner.ctx_order.append(i)

                output, selected_idx = select_confident_samples(output, args.selection_p)
                loss = avg_entropy(output).mean()

                self.optimizer.zero_grad()
                # compute gradient and do SGD step
                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss = torch.tensor(0.0).cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                exactnumber = exactnumber.reshape(-1)
                output = self.model(image, exactnumber + 1)
                output = output.view(output.size()[0], exactnumber.reshape(-1).shape[0], -1)
                output = output.mean(1)

        return_dict = {
            'output':output,
        }

        return return_dict



