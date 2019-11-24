import time
import torch
import torch.autograd as autograd
import torch.nn as nn


class FoollingModel(nn.Module):
    def __init__(self, layers, img_size=(224, 224), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.device = next(layers.parameters()).device
        self.img_size = img_size
        
    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.reshape((batch_size, 3, self.img_size[0], self.img_size[1]))
            return self.layers(x).reshape(batch_size, -1)
        else:
            return self.layers(x)

    def single_get_matvecs(self, img_batch):
        def _get_matvecs(img, v1=None, v2=None):
            assert v1 is not None or v2 is not None
            img_copy = torch.empty_like(img).copy_(img).unsqueeze(0).requires_grad_(True)
            output = self.forward(img_copy).squeeze(0)
            if v1 is None:
                v1 = torch.zeros_like(img_copy.squeeze(0), device=self.device)
                v2 = torch.empty_like(v2, device=self.device).copy_(v2).requires_grad_(True)
            if v2 is None:
                v1 = torch.empty_like(v1, device=self.device).copy_(v1)
                v2 = torch.zeros_like(output, device=self.device).requires_grad_(True)
            J_T_v2 = autograd.grad(v2.T @ output, img_copy, create_graph=True)[0].squeeze(0)
            g = J_T_v2.T @ v1
            v2.data = torch.zeros_like(v2)
            J_v1 = autograd.grad(g, v2)[0]
            return J_v1.detach(), J_T_v2.detach()
        
        def matvec(v1):
            return sum([
                _get_matvecs(img.flatten(), v1=v1)[0]
                for img in img_batch
            ])
            
        def matvec_T(v2):
            return sum([
                _get_matvecs(img.flatten(), v2=v2)[1]
                for img in img_batch
            ])
        
        return matvec, matvec_T
