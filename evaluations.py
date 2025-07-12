import torch
from torcheval.metrics.metric import Metric
from torcheval.metrics import PeakSignalNoiseRatio
from torcheval.metrics import MeanSquaredError

class PSNR():
    def __init__(self, device='cpu'):
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.device = device
    
    def evaluate(self, imgs1, imgs2):
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        
        self.psnr.update(imgs1, imgs2)
        
        return self.psnr.compute().item()
# end of class

class MSE():
    def __init__(self, device='cpu'):
        self.mse = MeanSquaredError().to(device)
        self.device = device
    
    def evaluate(self, imgs1, imgs2):
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        
        if imgs1.dim() > 2:
            imgs1 = imgs1.view(imgs1.shape[0], -1)
            imgs2 = imgs2.view(imgs1.shape[0], -1)
        
        self.mse.update(imgs1, imgs2)
        
        return self.mse.compute().item()
# end of class