import torch
import numpy as np
from pathlib import Path

patch_path = Path('training/sentinel_patches/atlantic_20231003.npy')
arr = np.load(patch_path)
print('Loaded', patch_path, 'shape', arr.shape, 'dtype', arr.dtype)

# B01 is band 5 index in the current stack, B03 is band 1
def tss(x):
    x = x.to(device)
    mask = ~torch.isnan(x)
    if mask.sum() == 0:
        return None, None, None, None, None
    x_valid = x[mask]
    p2 = torch.quantile(x_valid, 0.02)
    p98 = torch.quantile(x_valid, 0.98)
    st = torch.clamp((x - p2) / (p98 - p2), 0.0, 1.0)
    nan_count = torch.isnan(st).sum().item()
    valid = st[~torch.isnan(st)]
    stats = {
        'p2': float(p2.item()),
        'p98': float(p98.item()),
        'min': float(valid.min().item()),
        'max': float(valid.max().item()),
        'mean': float(valid.mean().item()),
        'std': float(valid.std().item()),
        'nan_count': nan_count,
    }
    return st, stats, p2, p98, mask

b01 = arr[5].astype(np.float32)
b03 = arr[1].astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device', device, 'cuda_available', torch.cuda.is_available())

b01_t = torch.from_numpy(b01)
b03_t = torch.from_numpy(b03)

st_b01, stats_b01, p2_b01, p98_b01, mask_b01 = tss(b01_t)
st_b03, stats_b03, p2_b03, p98_b03, mask_b03 = tss(b03_t)

ratio = None
if st_b01 is not None and st_b03 is not None:
    with torch.no_grad():
        denom = torch.where(st_b03 == 0, torch.tensor(1e-6, device=device), st_b03)
        ratio = st_b01 / denom

out = {
    'b01': stats_b01,
    'b03': stats_b03,
    'b01_b03_ratio': float(torch.nanmean(ratio).item()) if ratio is not None else None,
    'ratio_nan': int(torch.isnan(ratio).sum().item()) if ratio is not None else None,
}
print('Metrics:', out)

out_path = Path('training/sentinel_patches/atlantic_20231003_contrast.npy')
np.save(out_path, st_b03.cpu().numpy() if st_b03 is not None else np.array([]))
print('Saved B03 contrast map to', out_path)
torch.save({'b01': st_b01.cpu() if st_b01 is not None else None, 'b03': st_b03.cpu() if st_b03 is not None else None, 'ratio': ratio.cpu() if ratio is not None else None}, Path('training/sentinel_patches/atlantic_20231003_contrast_torch.pt'))
print('Saved torch contrast map to training/sentinel_patches/atlantic_20231003_contrast_torch.pt')
