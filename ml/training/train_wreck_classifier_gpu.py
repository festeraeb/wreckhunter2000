#!/usr/bin/env python3
"""Train a PyTorch-based classifier on wreck DNA library using GPU."""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / 'wreck_dna_library.csv'
OUT_MODEL = ROOT / 'wreck_classifier_v1_pytorch.pt'

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

# Diagnostic helper
print('Python interpreter:', sys.executable)
print('Python version:', sys.version.replace('\n', ' '))

try:
    import torch
    print('Torch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA devices:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('CUDA device name:', torch.cuda.get_device_name(0))
except ImportError as e:
    raise RuntimeError('Missing required package torch. Install via conda or pip in the active environment.') from e

df = pd.read_csv(CSV_PATH)
label_map = {'Cedarville':1,'Acme':1,'Sandusky':0,'Atlantic':0,'Sport':1,'Philadelphia':1}
df['label']=df['name'].map(label_map).fillna(0).astype(int)

df['b01b03_med']=df['b01b03_med'].fillna(0.0)
df['fog_pulse']=df['fog_pulse'].astype(int)
df['sdb_deviation']=df['sdb_deviation'].astype(int)
df['ecostress_zscore']=df['ecostress_zscore'].fillna(0.0)
df['sar_coherence']=df['sar_coherence'].fillna(0.0)

features=['sdb_delta_m','b01b03_med','fog_pulse','sdb_deviation','ecostress_zscore','sar_coherence']
X=df[features].astype(np.float32).values
y=df['label'].astype(np.int64).values

X=torch.tensor(X)
y=torch.tensor(y)

dataset=TensorDataset(X,y)
loader=DataLoader(dataset,batch_size=2,shuffle=True)

device='cuda' if torch.cuda.is_available() else 'cpu'
print('Using device', device)
print('torch version', torch.__version__)
print('torch cuda available', torch.cuda.is_available())
print('torch cuda version', torch.version.cuda)
try:
    print('torch cuda devices', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('cuda device name', torch.cuda.get_device_name(0))
        print('cuda device properties', torch.cuda.get_device_properties(0))
except Exception as e:
    print('cuda debug info failed:', e)

model=nn.Sequential(
    nn.Linear(len(features),32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,2)
).to(device)

opt=torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()

for epoch in range(15):
    total_loss=0
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        opt.zero_grad()
        out=model(xb)
        loss=loss_fn(out,yb)
        loss.backward(); opt.step()
        total_loss+=loss.item()
    print(f'Epoch {epoch+1} loss {total_loss/len(loader):.4f}')

torch.save(model.state_dict(),OUT_MODEL)
print('Saved',OUT_MODEL)
