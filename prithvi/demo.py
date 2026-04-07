import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from prithvi_mae import PrithviMAE 

#setup and path 
folders = ["march", "june", "august", "nov"]
#band order: B, G, R, NIR, SWIR1, SWIR2
bands_to_load = ["B02", "B03", "B04", "B08", "B11", "B12"] 

# data loading 
all_frames = []
for m in folders:
    month_bands = []
    for b in bands_to_load:
        #find the file that contains the band name 
        path = list(Path(m).glob(f"*{b}*"))[0] 
        with rasterio.open(path) as src:
            month_bands.append(src.read(1).astype(np.float32) / 10000.0)
    all_frames.append(np.stack(month_bands))

stack = np.stack(all_frames) # Shape: (4, 6, H, W)


#NDVI = (NIR - Red) / (NIR + Red)
def get_ndvi(frame):
    red = frame[2]
    nir = frame[3]
    return (nir - red) / (nir + red + 1e-10)

ndvi_march = get_ndvi(stack[0])
ndvi_aug   = get_ndvi(stack[2])
ndvi_nov   = get_ndvi(stack[3])

# Does it look like a double crop?
is_double_crop = (ndvi_march.mean() > 0.4) and (ndvi_aug.mean() < 0.3) and (ndvi_nov.mean() > 0.4)
print(f"Double Crop Detected: {is_double_crop}")



# model loading
model = PrithviMAE(
    img_size=224, patch_size=(1, 16, 16), num_frames=4, 
    in_chans=6, embed_dim=1024, encoder_only=True
)

# Load the weights 
weights = torch.load("prithvi_300m_tl/Prithvi_EO_V2_300M_TL.pt", map_location="cpu")
model.load_state_dict(weights.get("model", weights), strict=False)

# extract features (inference)
input_tensor = torch.from_numpy(stack).float().unsqueeze(0)
with torch.no_grad():
    features = model.forward_features(input_tensor)

    print(f"Foundation Model Features Shape: {features[-1].shape}")

#plot
plt.imshow(ndvi_march, cmap="RdYlGn")
plt.title("March NDVI - Basic Check")
plt.show()