import torch

class CLIPZS():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    def prepare_model_and_optimization(self, args):
        self.model.eval()

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = self.model(images)

        return_dict = {
            'output':output,
        }

        return return_dict

