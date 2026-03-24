import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image

plt.switch_backend("agg")  # avoid main thread issues

def sphere_plot_tensor(embeddings, labels, figsize=(10, 10), dpi=100):
    """
    Create a 3D sphere plot of embeddings and return as a torch tensor.

    Args:
        embeddings: (N, 3) array of points on the unit sphere.
        labels: (N,) array of labels for coloring.
        figsize: tuple (width, height) in inches.
        dpi: resolution.

    Returns:
        torch.Tensor of shape (3, H, W) with RGB values in [0,1].
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere surface
    r = 1
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)

    # Plot points
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
               c=labels, s=20, marker='.')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.tight_layout()

    # Save figure to in‑memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Convert buffer to tensor
    image = ToTensor()(Image.open(buf))
    plt.close(fig)
    return image