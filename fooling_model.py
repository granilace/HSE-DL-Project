import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from utils import for_imagenet


class BaseFoolingModel(nn.Module):
    def __init__(self, layers, img_size):
        super().__init__()
        self.layers = layers
        for layer in self.layers.parameters():
            layer.requires_grad = False
        self.img_size = img_size
        
    def full_forward(self, x):
        return self.layers(x)
        
    def forward(self, x, n_layer):
        if x.dim() == 1:
            x = x.reshape((3, self.img_size[0], self.img_size[1]))
        if x.dim() == 3:
            x = for_imagenet(x)
            x = x.reshape((1, 3, self.img_size[0], self.img_size[1]))
        else:
            raise ValueError("expected single image")
        return self.layers[:n_layer](x).flatten()

    def get_matvecs(self, img_batch, n_layer):
        device = img_batch.device
        input_size = np.prod(img_batch.shape[1:])
        batch_size = img_batch.shape[0]
        output_size = np.prod(self.forward(img_batch[0], n_layer).shape)
        
        def _get_matvecs(img, _v1=None, _v2=None):
            assert _v1 is not None or _v2 is not None
            assert _v1 is None or _v2 is None
            img_copy = img.view(-1).requires_grad_(True)
            output = self.forward(img_copy, n_layer)
            v1 = torch.zeros(input_size, device=device)
            if _v1 is not None:
                v1.copy_(_v1)
            v2 = torch.zeros(output_size, device=device)
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

    
class BaseVGGFoolingModel(BaseFoolingModel):    
    def __init__(self, model, img_size=(224, 224)):
        layers = nn.Sequential(
            *model.features,
            model.avgpool,
            nn.Flatten(1),
            *model.classifier
        )
        super().__init__(layers, img_size)


class VGG16FoolingModel(BaseVGGFoolingModel):
    LAYERS_IDS = [
        ('block2_conv1', 6),
        ('block2_conv2', 8),
        ('block2_pool', 10),
        ('block3_conv1', 11),
        ('block3_conv2', 13),
        ('block3_conv3', 15),
        ('block3_pool', 17)
    ]


class VGG19FoolingModel(BaseVGGFoolingModel):
    LAYERS_IDS = [
        ('block2_conv1', 6),
        ('block2_conv2', 8),
        ('block2_pool', 10),
        ('block3_conv1', 11),
        ('block3_conv2', 13),
        ('block3_conv3', 15),
        ('block3_conv4', 17),
        ('block3_pool', 19)
    ]


class BaseResNetFoolingModel(BaseFoolingModel):
    def __init__(self, model, img_size=(224, 224)):
        layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            *model.layer1,
            *model.layer2,
            *model.layer3,
            *model.layer4,
            model.avgpool,
            nn.Flatten(start_dim=1),
            model.fc
        )
        super().__init__(layers, img_size)


class ResNet50FoolingModel(BaseResNetFoolingModel):
    LAYERS_IDS = (
        ('conv1', 1),
        ('pool1', 4),
        ('block1', 5),
        ('block2', 6),
        ('block3', 7),
        ('block4', 8)
    )
