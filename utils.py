import numpy as np
import torch

from torchvision import transforms


imagenet_normalization = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)
to_255 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0] * 3, [1.0/255])
])


def for_imagenet(image):  # from 255 to net's expected range
    if image.dim() == 3:
        return imagenet_normalization(image / 255.0)
    elif image.dim() == 4:
        return torch.stack([for_imagenet(img) for img in image])
    else:
        raise ValueError('expected image of dim 3 or 4')

def psi(x, p):
    return torch.sign(x) * torch.abs(x) ** (p - 1) if p != 1 else torch.sign(x)

def conj(p):
    return float(p) / (p - 1) if p != np.inf else 1.0

def power_method(init, matvec, matvec_T, p=np.inf, q=10, max_iter=20):
    x = init / torch.norm(init, p)
    s = torch.norm(matvec(x), q)
    p_conj = conj(p)
    for i in range(max_iter):
        S_x = psi(matvec_T(psi(matvec(x), q)), p_conj)
        x = S_x / torch.norm(S_x, p)
    s = torch.norm(matvec(x), q)
    return x, s.item()

def process_image(img):
    img = (img - img.min()) / (img.max() - img.min())
    return torch.clamp(img.cpu().permute(1, 2, 0), 0, 1)

def calculate_fooling_rate(model, layer_id, train_batch, val_data, perturb_norm, p=np.inf, q=10.0, max_iter=20):
    matvec, matvec_T = model.get_matvecs(train_batch, layer_id)
    perturb, _ = power_method(
        init=torch.rand(3 * 224 * 224, device=train_batch.device) - 0.5,
        matvec=matvec, 
        matvec_T=matvec_T,
        p=p,
        q=q,
        max_iter=max_iter
    )
    normed_perturb = perturb.view(3, 224, 224) / torch.norm(perturb, p) * perturb_norm
    
    count_same = 0
    count_diff = 0
    inference_bs = 128
    for i in range(len(val_data) // inference_bs):
        val_batch = val_data[i * inference_bs: (i + 1) * inference_bs]
        with torch.no_grad():
            orig_output = model.full_forward(for_imagenet(val_batch))
            perturb_output = model.full_forward(for_imagenet(val_batch + normed_perturb))
        _, orig_preds = torch.softmax(orig_output, 1).max(1)
        _, perturb_preds = torch.softmax(perturb_output, 1).max(1)
        
        count_same += (perturb_preds == orig_preds).sum().item()
        count_diff += (perturb_preds != orig_preds).sum().item()

    return float(count_diff) / (count_diff + count_same), normed_perturb.cpu()

