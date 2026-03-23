import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import load_temporal_stack

BASE_DIR = Path(__file__).resolve().parent
FOLDERS  = ["march", "june", "august", "nov"]
LABELS   = ["March", "June", "August", "November"]

#load data 
spectral, ndvi = load_temporal_stack(BASE_DIR, FOLDERS)

# ndvi shape
ndvi_march, ndvi_june, ndvi_aug, ndvi_nov = ndvi[0], ndvi[1], ndvi[2], ndvi[3]

#cloud mask
# clouds = high reflectance across all visible bands
blue_june = spectral[1, 0]  # B02 june
cloud_mask = blue_june > 0.3  # high blue reflectance = likely cloud
print(f"Cloud coverage June: {cloud_mask.mean()*100:.1f}%")

#double cropping rules
#high vegetation in spring and high vegetation in autumn
# with a clear dip in between (harvest in summer)

# Single crop: peaks once (usually summer)
# Double crop: peaks twice (spring + autumn), dips in summer

spring_green   = ndvi_march > 0.45
summer_harvest = ndvi_aug   < 0.30         # stricter harvest dip
autumn_regrow  = ndvi_nov   > 0.40         # stricter regrowth
not_cloud      = ~cloud_mask               # exclude cloudy pixels

double_crop = (spring_green & summer_harvest & autumn_regrow & not_cloud).astype(np.float32)

soft_map = (
    (ndvi_march > 0.45).astype(float) +
    (ndvi_june  > 0.5).astype(float)  * (~cloud_mask).astype(float) +  # cloud weighted
    (ndvi_aug   < 0.30).astype(float) +
    (ndvi_nov   > 0.40).astype(float)
) / 4.0

soft_map = np.where(cloud_mask, np.nan, soft_map)

print(f"Double crop area (hard): {double_crop.mean()*100:.1f}% of patch")
print(f"Mean soft probability:   {soft_map.mean():.3f}")

#visualize 
fig = plt.figure(figsize=(18, 12))

# row 1: RGB
for t in range(4):
    ax = fig.add_subplot(3, 4, t + 1)
    rgb = spectral[t][[2, 1, 0]].transpose(1, 2, 0)
    ax.imshow(np.clip(rgb * 3.5, 0, 1))
    ax.set_title(f"RGB {LABELS[t]}"); ax.axis("off")

# row 2: NDVI per month
for t in range(4):
    ax = fig.add_subplot(3, 4, t + 5)
    ax.imshow(ndvi[t], cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title(f"NDVI {LABELS[t]} (μ={ndvi[t].mean():.2f})"); ax.axis("off")

# row 3: double crop maps
ax1 = fig.add_subplot(3, 2, 5)
ax1.imshow(double_crop, cmap="RdYlGn", vmin=0, vmax=1)
ax1.set_title(f"Double Crop Map (Hard Rules)\n{double_crop.mean()*100:.1f}% flagged")
ax1.axis("off")

ax2 = fig.add_subplot(3, 2, 6)
im = ax2.imshow(soft_map, cmap="magma", vmin=0, vmax=1)
ax2.set_title("Double Crop Probability (Soft Score)")
ax2.axis("off")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

plt.suptitle("Double Cropping Detection — NDVI Temporal Rules", fontsize=14)
plt.tight_layout()
plt.savefig(BASE_DIR / "ndvi_crop_map.png")
plt.show(block=True)

from data_loader import load_temporal_stack, calculate_area_stats



#Hard Rules to create the binary mask
double_crop_mask = (spring_green & summer_harvest & autumn_regrow & not_cloud)

#Calculate Hectares
ha_total, pixels = calculate_area_stats(double_crop_mask)

#Confidence Score

agreement_score = np.mean(soft_map[double_crop_mask == 1]) if pixels > 0 else 0

print("-" * 30)
print(f"Summary")
print(f"Total Pixels Flagged: {pixels}")
print(f"Total Double Crop Area: {ha_total:.2f} ha")
print(f"Foundation Model Confidence: {agreement_score:.2f}")
print("-" * 30)