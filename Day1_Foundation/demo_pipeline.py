import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt

with rasterio.open("b02.tiff") as src:
    blue = src.read(1).astype(float)

with rasterio.open("b03.tiff") as src:
    green = src.read(1).astype(float)

with rasterio.open("b04.tiff") as src:
    red = src.read(1).astype(float)

with rasterio.open("b08.tiff") as src:
    nir = src.read(1).astype(float)

with rasterio.open("b11.tiff") as src:
    swir1 = src.read(1).astype(float)

with rasterio.open("b12.tiff") as src:
    swir2 = src.read(1).astype(float)

print("Bands loaded")

#stack bands into a single array

image = np.stack([blue, green, red, nir, swir1, swir2])

print("Image shape:", image.shape)


#covert to tensor
image_tensor = torch.tensor(image).float()
image_tensor = image_tensor.unsqueeze(0)

print("Tensor shape:", image_tensor.shape)



#create rgb image for visualization

rgb = np.stack([red, green, blue], axis=2)

p2, p98 = np.percentile(rgb, (2, 98))
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

plt.figure(figsize=(8,6))
plt.imshow(rgb)
plt.title("Sentinel-2 RGB Image")
plt.axis("off")
plt.show()

#compute ndvi

ndvi = (nir - red) / (nir + red + 1e-10)

plt.figure(figsize=(8,6))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar()
plt.title("NDVI Map")
plt.axis("off")
plt.show()




features = image_tensor.flatten()

print("Feature vector size:", features.shape)

plot_features = features[:1000].numpy()

plt.figure(figsize=(10,4))
plt.plot(plot_features)
plt.title("Feature Embeddings (First 1000 values)")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.show()