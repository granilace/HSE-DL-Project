import torch
from PIL import Image
from utils import to_255


class ImagenetDataset:
    def __init__(self, paths, device):
        self.paths = paths
        self.device = device
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return torch.stack([
                self[i] for i in range(key.start or 0, key.stop or len(self.paths), key.step or 1)
            ])
        img = Image.open(self.paths[key]).convert('RGB')
        return to_255(img).to(self.device)
