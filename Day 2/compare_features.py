import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.models as models



def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(float)

blue = load_band("b02.tiff")
green = load_band("b03.tiff")
red = load_band("b04.tiff")
nir = load_band("b08.tiff")
swir1 = load_band("b11.tiff")
swir2 = load_band("b12.tiff")

print("Bands loaded successfully")


image = np.stack([blue, green, red, nir, swir1, swir2])
print("Image shape:", image.shape)

#rgb 
rgb = np.stack([red, green, blue], axis=2)

p2, p98 = np.percentile(rgb, (2, 98))
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

plt.figure(figsize=(8,6))
plt.imshow(rgb)
plt.title("Sentinel-2 RGB Image")
plt.axis("off")
plt.show()

#ndvi

ndvi = (nir - red) / (nir + red + 1e-10)

ndvi_mean = np.mean(ndvi)
ndvi_std = np.std(ndvi)

print("\nNDVI Features:")
print("Mean:", ndvi_mean)
print("Std:", ndvi_std)

plt.figure(figsize=(8,6))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar()
plt.title("NDVI Map")
plt.axis("off")
plt.show()

#patch extraction

patch = image[:, :224, :224]

# normalize 
patch = patch / np.max(patch)

patch_tensor = torch.tensor(patch).float().unsqueeze(0)

print("\nPatch shape:", patch_tensor.shape)


#convert to 3-channel 


rgb_patch = patch[[2,1,0], :, :]  # R,G,B

rgb_tensor = torch.tensor(rgb_patch).float().unsqueeze(0)

#load ViT model

model = models.vit_b_16(pretrained=True)
model.eval()

# get embeddings
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

#extract features

with torch.no_grad():
    features = feature_extractor(rgb_tensor)

features = features.squeeze()

print("\nModel Feature Vector Shape:", features.shape)

#compare features
print("\n Comparison")
print("NDVI Mean:", ndvi_mean)
print("NDVI Std:", ndvi_std)
print("Model Feature Size:", features.shape)

# plot first 100 features
plt.figure(figsize=(10,4))
plt.plot(features[:100].numpy())
plt.title("Model Feature Embeddings (First 100)")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.show()

#double cropping 

# second patch from different location
patch2 = image[:, 300:524, 300:524]

nir2 = patch2[3]
red2 = patch2[2]

ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-10)


print("Patch1 NDVI mean:", ndvi_mean)
print("Patch2 NDVI mean:", np.mean(ndvi2))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(ndvi, cmap="RdYlGn")
plt.title("Patch 1 NDVI")

plt.subplot(1,2,2)
plt.imshow(ndvi2, cmap="RdYlGn")
plt.title("Patch 2 NDVI")

plt.show()