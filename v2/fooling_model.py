import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn

from utils import for_imagenet


class FoollingModel(nn.Module):
    def __init__(self, layers, img_size=(224, 224), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.device = next(layers.parameters()).device
        self.img_size = img_size
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape((3, self.img_size[0], self.img_size[1]))
        if x.dim() == 3:
            x = for_imagenet(x)
            x = x.reshape((1, 3, self.img_size[0], self.img_size[1]))
        else:
            raise ValueError("expected single image")
        return self.layers(x).flatten()

    def get_matvecs(self, img_batch):
        input_size = np.prod(img_batch.shape[1:])
        batch_size = img_batch.shape[0]
        output_size = np.prod(self.forward(img_batch[0]).shape)
        
        def _get_matvecs(img, _v1=None, _v2=None):
            assert _v1 is not None or _v2 is not None
            assert _v1 is None or _v2 is None
            img_copy = img.view(-1).requires_grad_(True)
            output = self.forward(img_copy)
            v1 = torch.zeros(input_size, device=self.device)
            if _v1 is not None:
                v1.copy_(_v1)
            v2 = torch.zeros(output_size, device=self.device)
            if _v2 is not None:
                v2.copy_(_v2)
            v2.requires_grad_(True)
            J_T_v2 = autograd.grad(v2 @ output, img_copy, create_graph=True)[0]
            J_v1 = autograd.grad(J_T_v2 @ v1, v2)[0]
            return J_v1.detach(), J_T_v2.detach()
        
        def matvec(v1):
            return torch.stack([
                _get_matvecs(img, _v1=v1)[0]
                for img in img_batch
            ])
            
        def matvec_T(v2):
            return sum([
                _get_matvecs(img.flatten(), _v2=v2)[1]
                for img, v2 in zip(img_batch, v2)
            ])
        
        return matvec, matvec_T
