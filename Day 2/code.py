import rasterio
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


try:
    with rasterio.open(os.path.join(os.path.dirname(__file__), "b02.tiff")) as src:
        blue = src.read(1).astype(float)
except FileNotFoundError:
    print("Error: File 'b02.tiff' not found. Please ensure the TIFF files are in the same directory as this script.")
    raise
except Exception as e:
    print(f"Error loading 'b02.tiff': {e}")
    raise

try:
    with rasterio.open(os.path.join(os.path.dirname(__file__), "b03.tiff")) as src:
        green = src.read(1).astype(float)
except FileNotFoundError:
    print("Error: File 'b03.tiff' not found.")
    raise
except Exception as e:
    print(f"Error loading 'b03.tiff': {e}")
    raise

try:
    with rasterio.open(os.path.join(os.path.dirname(__file__), "b04.tiff")) as src:
        red = src.read(1).astype(float)
except FileNotFoundError:
    print("Error: File 'b04.tiff' not found.")
    raise
except Exception as e:
    print(f"Error loading 'b04.tiff': {e}")
    raise

try:
    with rasterio.open(os.path.join(os.path.dirname(__file__), "b08.tiff")) as src:
        nir = src.read(1).astype(float)
except FileNotFoundError:
    print("Error: File 'b08.tiff' not found.")
    raise
except Exception as e:
    print(f"Error loading 'b08.tiff': {e}")
    raise

try:
    with rasterio.open(os.path.join(os.path.dirname(__file__), "b11.tiff")) as src:
        swir1 = src.read(1).astype(float)
except FileNotFoundError:
    print("Error: File 'b11.tiff' not found.")
    raise
except Exception as e:
    print(f"Error loading 'b11.tiff': {e}")
    raise

try:
    with rasterio.open(os.path.join(os.path.dirname(__file__), "b12.tiff")) as src:
        swir2 = src.read(1).astype(float)
except FileNotFoundError:
    print("Error: File 'b12.tiff' not found.")
    raise
except Exception as e:
    print(f"Error loading 'b12.tiff': {e}")
    raise

print("Bands loaded")

#convert to tensor
image = np.stack([blue, green, red, nir, swir1, swir2])
print("Image shape:", image.shape)

image_tensor = torch.tensor(image).float()
image_tensor = image_tensor.unsqueeze(0)
print("Tensor shape:", image_tensor.shape)

# visualize RGB
rgb = np.stack([red, green, blue], axis=2)
p2, p98 = np.percentile(rgb, (2, 98))
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

plt.figure(figsize=(8,6))
plt.imshow(rgb)
plt.title("Sentinel-2 RGB Image")
plt.axis("off")
plt.savefig(os.path.join(os.path.dirname(__file__), 'sentinel_rgb.png'))

# 4. compute NDVI
ndvi = np.nan_to_num((nir - red) / (nir + red + 1e-10))
ndvi_mean= np.mean(ndvi)
ndvi_std = np.std(ndvi)

print("Mean NDVI:", ndvi_mean)
print("NDVI Std Dev:", ndvi_std)   

plt.figure(figsize=(8,6))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar()
plt.title("NDVI Map")
plt.axis("off")
plt.show()

# Extract features
features = torch.cat([
           image_tensor.mean(dim=(2,3)), # average reflectance per band 
           torch.tensor([[ndvi_mean, ndvi_std]], dtype=torch.float32)  
], dim=1)   

print("Feature vector size:", features.shape)

# Plot Features 

plot_features = features.squeeze().numpy() 

feature_names = ['Blue Mean', 'Green Mean', 'Red Mean', 'NIR Mean', 
                 'SWIR1 Mean', 'SWIR2 Mean', 'NDVI Mean', 'NDVI Std']

plt.figure(figsize=(10,5))
plt.bar(feature_names, plot_features, color='teal')
plt.title("Feature Embeddings (Band Means & NDVI Stats)")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.xticks(rotation=45, ha='right') 
plt.tight_layout() 
plt.show()