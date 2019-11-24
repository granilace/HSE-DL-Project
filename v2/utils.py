import torch

def psi(x, r):
    return torch.sign(x) * torch.pow(torch.abs(x), r - 1)

def conj(p):
    return float(p) / (p - 1) if p != float('inf') else 1.0

def power_method(init, matvec, matvec_T, p=float('inf'), q=10, tol=1e-2, max_iter=100):
    x = init / torch.norm(init, p)
    s = torch.norm(matvec(x), q)
    p_conj = conj(p)
    prev_x = x.clone()
    for i in range(max_iter):
        S_x = psi(matvec_T(psi(matvec(x), q)), p_conj)
        x = S_x / torch.norm(S_x, p)
        print('delta', torch.norm(prev_x - x))
        if torch.norm(prev_x - x) < tol:
            break
        prev_x = x
    s = torch.norm(matvec(x), q)
    return x, s

def process_image(img):
    img = (img - img.min()) / (img.max() - img.min())
    return torch.clamp(img.cpu().permute(1, 2, 0), 0, 1)