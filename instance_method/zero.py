import torch

def break_sample_tie(ties, logit, device):
    # ties = torch.tensor(ties, dtype=torch.int, device=device)
    # logit[~ties] = -torch.inf

    mask = torch.zeros_like(logit, dtype=torch.bool, device=device)
    logit[~mask] = -torch.inf
    scalar_pred = torch.argmax(logit, dim=-1)
    return scalar_pred

def greedy_break(ties, logits, device):
    ties_tensor = torch.tensor(ties, dtype=torch.int, device=device)
    preds = torch.argmax(logits, dim=1)
    for pred in preds:
        if pred in ties_tensor:
            return pred
    return break_sample_tie(ties, logit=logits[0], device=device)

def confidence_filter(logits: torch.Tensor, probs: torch.Tensor, top:float, return_idx: bool=False):
    batch_entropy = -(probs * probs.log()).sum(1)
    full_idx = torch.argsort(batch_entropy, descending=False)
    filt_idx = full_idx[:int(batch_entropy.size()[0] * top)]
    if not return_idx:
        return logits[filt_idx]
    return logits[filt_idx], filt_idx, full_idx

class ZERO():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gamma = 0.3

    def prepare_model_and_optimization(self, args):
        self.model.eval()

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):
        
        with torch.no_grad():
            # with torch.cuda.amp.autocast(): # do not use amp for ZERO
            l = self.model(images)

        with torch.no_grad():
            p = l.softmax(1)
            l_filt, _, sorted_idx = confidence_filter(l, p, top=self.gamma, return_idx=True)
            zero_temp = torch.finfo(l_filt.dtype).eps
            p_bar = (l_filt / zero_temp).softmax(1).sum(0)

            max_counts, scalar_pred = torch.max(p_bar, dim=-1)
            ties = [scalar_pred.item()]
            for i in range(len(p_bar)):
                if i == scalar_pred: continue
                if p_bar[i] == max_counts: ties.append(i)

            if len(ties) > 1:
                k = int(images.size(0) * self.gamma)
                sorted_l = l[sorted_idx]
                scalar_pred = greedy_break(ties, sorted_l[k:], device=self.device)
                p_bar[scalar_pred]+=1

        return_dict = {
            'output':p_bar.unsqueeze(0),
        }

        return return_dict


