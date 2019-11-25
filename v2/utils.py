import numpy as np
import torch

def psi(x, p):
    return torch.sign(x) * torch.abs(x) ** (p - 1) if p != 1 else torch.sign(x)

def conj(p):
    return float(p) / (p - 1) if p != float('inf') else 1.0

def power_method(init, matvec, matvec_T, p=float('inf'), q=10, tol=1e-2, max_iter=20):
    x = init / torch.norm(init, p)
    s = torch.norm(matvec(x), q)
    p_conj = conj(p)
    prev_x = x.clone()
    for i in range(max_iter):
        S_x = psi(matvec_T(psi(matvec(x), q)), p_conj)
        x = S_x / torch.norm(S_x, p)
        if torch.norm(prev_x - x, p) < tol:
            break
        prev_x = x
    return x, s

def process_image(img):
    img = (img - img.min()) / (img.max() - img.min())
    return torch.clamp(img.cpu().permute(1, 2, 0), 0, 1)
