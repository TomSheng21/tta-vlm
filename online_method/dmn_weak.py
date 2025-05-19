import torch

class MemoryBank:
    def __init__(self, text_fea, C, max_len, D, beta, device='cpu'):
        self.C = C
        self.max_len = max_len
        self.D = D
        self.device = device
        self.text_fea = text_fea
        self.beta = beta
        
        self.memory = torch.zeros((C, max_len, D), device=device)
        self.ent_mem = torch.zeros((C, max_len), device=device)
        self.counts = torch.zeros(C, dtype=torch.long, device=device)
        
    def update(self, feature, pl, ent):
        assert 0 <= pl < self.C
        assert feature.shape[-1] == self.D
        
        current_count = self.counts[pl]
        
        if current_count < self.max_len:
            self.memory[pl, current_count] = feature
            self.ent_mem[pl, current_count] = ent
            self.counts[pl] += 1
        else:
            replace_idx = torch.sort(self.ent_mem[pl], descending=True)[1][0]
            self.memory[pl, replace_idx] = feature
            self.ent_mem[pl, replace_idx] = ent
    
    def reset(self):
        self.memory.zero_()
        self.ent_mem.zero_()
        self.counts.zero_()

    def read_out(self, feature):
        assert feature.shape[-1] == self.D

        Cd = None
        
        for i in range(self.C):
            if self.counts[i] > 0:
                My = torch.cat([self.memory[i, :self.counts[i]], self.text_fea[i].unsqueeze(0)], dim=0)
            else:
                My = self.text_fea[i].unsqueeze(0)

            My = My / torch.norm(My, dim=1, keepdim=True)
            feature = feature / torch.norm(feature, dim=1, keepdim=True)
            sim = feature @ My.T
            wei = torch.exp(-self.beta*(1 - sim))

            Cy = wei @ My
            Cy = Cy / torch.norm(Cy, dim=1, keepdim=True)

            if Cd is None:
                Cd = Cy
            else:
                Cd = torch.cat([Cd, Cy], dim=0)

        return Cd
  

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class DMN_WEAK():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        self.beta = 5.5
        self.alpha = 1.0
        self.max_len = 50

    def prepare_model_and_optimization(self, args):
        self.model.eval()
        text_features = self.model.get_text_features()
        self.mem = MemoryBank(text_features, text_features.shape[0], self.max_len, text_features.shape[1], self.beta, device=self.device)

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features, text_features, logit_scale = self.model.forward_features(images)
                logits = logit_scale * image_features @ text_features.t()
                softmax0 = logits.softmax(dim=-1)

            ent0 = Entropy(softmax0)
            pred0 = torch.max(logits, 1)[1].item()
            num_classes = logits.size(1)

            self.mem.update(image_features, pred0, ent0)
            Cd = self.mem.read_out(image_features)
            output_d = logit_scale * image_features @ Cd.t()
            softmax_d = output_d.softmax(dim=-1)

            final_pred = softmax0 + self.alpha * softmax_d

        return_dict = {
            'output':final_pred,
        }

        return return_dict



