import matplotlib
matplotlib.use("TkAgg")

import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
import os

DIR = os.path.dirname(__file__)

# --- Load Bands ---
def load_band(filename):
    with rasterio.open(os.path.join(DIR, filename)) as src:
        return src.read(1).astype(float)

blue, green, red, nir, swir1, swir2 = [load_band(f"b{b}.tiff") for b in ("02","03","04","08","11","12")]
print("Bands loaded. Image shape:", np.stack([blue, green, red, nir, swir1, swir2]).shape)

image = np.stack([blue, green, red, nir, swir1, swir2])

# --- RGB Visualization ---
rgb = np.stack([red, green, blue], axis=2)
p2, p98 = np.percentile(rgb, (2, 98))
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

plt.figure(figsize=(8, 6))
plt.imshow(rgb); plt.title("Sentinel-2 RGB Image"); plt.axis("off")
plt.savefig(os.path.join(DIR, "rgb_image.png"))
plt.show(); input("Press Enter to continue...")

# --- NDVI ---
ndvi = np.nan_to_num((nir - red) / (nir + red + 1e-10))
print(f"\nNDVI — Mean: {np.mean(ndvi):.4f}, Std: {np.std(ndvi):.4f}")

plt.figure(figsize=(8, 6))
plt.imshow(ndvi, cmap="RdYlGn"); plt.colorbar(); plt.title("NDVI Map"); plt.axis("off")
plt.savefig(os.path.join(DIR, "ndvi_map.png"))
plt.show(); input("Press Enter to continue...")

# --- ViT Feature Extraction ---
rgb_patch = (image[[2, 1, 0], :224, :224] / image.max())
rgb_tensor = torch.tensor(rgb_patch).float().unsqueeze(0)
print("\nPatch tensor shape:", rgb_tensor.shape)

model = models.vit_b_16(pretrained=True)
model.eval()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

with torch.no_grad():
    features = feature_extractor(rgb_tensor).squeeze()
print("Feature vector shape:", features.shape)

plt.figure(figsize=(10, 4))
plt.plot(features.flatten()[:100].numpy())
plt.title("ViT Feature Embeddings (First 100)"); plt.xlabel("Feature Index"); plt.ylabel("Value")
plt.savefig(os.path.join(DIR, "feature_embeddings.png"))
plt.show(); input("Press Enter to continue...")

# --- Double Cropping NDVI Comparison ---
def patch_ndvi(img, row, col):
    p = img[:, row:row+224, col:col+224]
    return np.nan_to_num((p[3] - p[2]) / (p[3] + p[2] + 1e-10))

ndvi1 = patch_ndvi(image, 0, 0)
ndvi2 = patch_ndvi(image, 300, 300)
print(f"\nPatch 1 NDVI mean: {np.mean(ndvi1):.4f} | Patch 2 NDVI mean: {np.mean(ndvi2):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in zip(axes, [ndvi1, ndvi2], ["Patch 1 NDVI", "Patch 2 NDVI"]):
    ax.imshow(data, cmap="RdYlGn"); ax.set_title(title)
plt.savefig(os.path.join(DIR, "ndvi_comparison.png"))
plt.show(); input("Press Enter to finish.")