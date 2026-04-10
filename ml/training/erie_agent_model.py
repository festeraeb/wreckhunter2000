"""Save current best_resnet18.pt as Lake Erie Central Basin agent."""
import torch
import os
import sys

BASE = r"C:\Users\thomf\programming\Bagrecovery"
src = os.path.join(BASE, "wreck_hunting_ml", "models", "best_resnet18.pt")
dst = os.path.join(BASE, "wreck_hunting_ml", "models", "erie_central_basin_agent.pt")

if not os.path.exists(src):
    print(f"ERROR: Source not found: {src}")
    sys.exit(1)

print(f"Loading {src} ...")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
print(f"  Epoch:  {ckpt.get('epoch', '?')}")
print(f"  Val acc: {ckpt.get('val_acc', '?')}")
print(f"  Phase:  {ckpt.get('phase', '?')}")
print(f"  FVD:    {ckpt.get('fvd_preprocessing', '?')}")
print(f"  Classes: {ckpt.get('class_names', '?')}")

# Tag with lake agent metadata
ckpt["agent_label"] = "Lake Erie Central Basin"
ckpt["lake"] = "Erie"
ckpt["basin"] = "central"
ckpt["description"] = (
    "ResNet-18 trained on Erie central basin aeromagnetic data "
    "(GSC lowalt 0.001deg), CRM v2 synthetic tiles, epoch 91, val_acc 0.8224"
)

torch.save(ckpt, dst)
size_mb = os.path.getsize(dst) / 1e6
print(f"\nSaved: {dst}")
print(f"Size:  {size_mb:.1f} MB")

# List all models
print("\n=== All models ===")
models_dir = os.path.join(BASE, "wreck_hunting_ml", "models")
for f in sorted(os.listdir(models_dir)):
    if f.endswith(".pt"):
        fp = os.path.join(models_dir, f)
        print(f"  {os.path.getsize(fp)/1e6:.1f} MB  {f}")
