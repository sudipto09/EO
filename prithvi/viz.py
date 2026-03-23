import matplotlib.pyplot as plt
import numpy as np

def plot_temporal_results(spectral_stack, ndvi_stack, crop_map, labels):
    fig = plt.figure(figsize=(16, 10))

    for t in range(4):
        # row 1: RGB
        ax = fig.add_subplot(3, 4, t + 1)
        rgb = spectral_stack[t][[2, 1, 0]].transpose(1, 2, 0)
        ax.imshow(np.clip(rgb * 3.5, 0, 1))
        ax.set_title(f"RGB {labels[t]}")
        ax.axis("off")

        # row 2: NDVI
        ax = fig.add_subplot(3, 4, t + 5)
        ax.imshow(ndvi_stack[t], cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"NDVI (μ={ndvi_stack[t].mean():.2f})")
        ax.axis("off")

    # row 3: crop probability map spanning full width
    ax_map = fig.add_subplot(3, 1, 3)
    im = ax_map.imshow(crop_map.squeeze(), cmap="magma")
    ax_map.set_title("Double Cropping Probability (Prithvi-EO Inference)")
    ax_map.axis("off")
    plt.colorbar(im, ax=ax_map, orientation="horizontal", fraction=0.046, pad=0.1)

    plt.tight_layout()
    plt.show(block=True)