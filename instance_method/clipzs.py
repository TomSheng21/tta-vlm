import torch

class CLIPZS():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def prepare_model_and_optimization(self, args):
        self.model.eval()

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                tta_output = self.model(image)

        return_dict = {
            'output':tta_output,
        }

        return return_dict

