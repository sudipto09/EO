import numpy as np

chip = np.load(r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output\prithvi_input_FID1_2024-05-13.npy')
print(chip.shape)   # (6, 224, 224)
print(chip.dtype)   # float32
print(chip.min(), chip.max())   

import numpy as np
import matplotlib.pyplot as plt

chip = np.load(r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output\prithvi_input_FID1_2024-05-13.npy')

plt.imshow(chip[3], cmap='RdYlGn')   # band index 3 = NIR, good for crop health
plt.colorbar()
plt.title('NIR band')
plt.show()

# Bands 2, 1, 0 = Red, Green, Blue in this stack
rgb = np.stack([chip[2], chip[1], chip[0]], axis=-1)

# Stretch to 0–1 for display
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.imshow(rgb)
plt.title('True colour')
plt.show()