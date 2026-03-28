"""Figure helpers for logging (W&B, etc.): PCA, sphere plots, mel spectrograms, confusion matrices."""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from src.utils.hydra_cfg import cfg_get, cfg_to_container

plt.switch_backend("agg")  # avoid RuntimeError: main thread is not in main loop


def mel_forward(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_inverse(m: np.ndarray | float) -> np.ndarray | float:
    return (10.0 ** (np.asarray(m) / 2595.0) - 1.0) * 700.0


def feature_plot_params_from_config(config: Any) -> dict[str, Any]:
    """
    Read ``transforms.instance_transforms.{train,inference}.get_feature`` and return
    parameters for logging **either** a mel spectrogram (torchaudio MelSpectrogram)
    **or** MFCCs (``MfccExtractor`` / ``mfcc`` in target).

    Returns:
        dict with ``kind`` ``\"mel\"`` or ``\"mfcc\"`` plus hop / rate / band counts.
    """
    raw = cfg_get(config, "transforms.instance_transforms.train.get_feature")
    if raw is None:
        raw = cfg_get(config, "transforms.instance_transforms.inference.get_feature")
    if raw is None:
        return {
            "kind": "mel",
            "hop_length": 160,
            "n_mels": 128,
            "sample_rate": 16000.0,
        }
    d = cfg_to_container(raw)
    if not isinstance(d, dict):
        d = {}
    target = str(d.get("_target_", "")).lower()
    if "mfcc" in target:
        sr = float(d.get("sample_rate", 16000))
        hop_sec = float(d.get("hop_length_sec", 0.01))
        return {
            "kind": "mfcc",
            "hop_length": int(sr * hop_sec),
            "sample_rate": sr,
            "n_mfcc": int(d.get("n_mfcc", 24)),
        }
    d2 = {k: v for k, v in d.items() if k != "_target_"}
    return {
        "kind": "mel",
        "hop_length": int(d2.get("hop_length", 160)),
        "n_mels": int(d2.get("n_mels", 128)),
        "sample_rate": float(d2.get("sample_rate", 16000)),
    }


def plot_images(
    imgs: torch.Tensor,
    names: list[str],
    figsize: tuple[float, float] = (12, 4),
) -> np.ndarray:
    """
    Combine several images into one figure (side by side).

    Args:
        imgs: (B, C, H, W).
        names: subplot titles; length must match ``B``.
        figsize: matplotlib figure size.

    Returns:
        (H, W, 3) float array in [0, 1] for ``wandb.Image``.
    """
    imgs = imgs.detach().cpu().float()
    b = imgs.shape[0]
    if b != len(names):
        raise ValueError(f"len(names) ({len(names)}) must match batch dim ({b})")
    fig, axes = plt.subplots(1, b, figsize=figsize)
    if b == 1:
        axes = [axes]
    for i in range(b):
        img = imgs[i].permute(1, 2, 0).numpy()
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(names[i])
        axes[i].axis("off")
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    out = ToTensor()(Image.open(buf)).permute(1, 2, 0).numpy()
    plt.close(fig)
    return out


def plot_mfcc_coeffs(
    mfcc: torch.Tensor,
    params: dict[str, Any] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (26, 7),
) -> np.ndarray:
    """
    Plot MFCC trajectories (linear coefficients, not log mel power) for W&B.

    Args:
        mfcc: (n_mfcc, time) or (C, n_mfcc, time); multi-channel is averaged.
        params: ``hop_length``, ``sample_rate``, ``n_mfcc`` from
            ``feature_plot_params_from_config`` (``kind`` ``\"mfcc\"``).
        title: optional figure title.

    Returns:
        (H, W, 3) float array in [0, 1].
    """
    spec = mfcc.detach().float().cpu().numpy()
    if spec.ndim == 3:
        spec = spec.mean(axis=0)
    ms = params or {}
    hop_length = int(ms.get("hop_length", 160))
    sample_rate = float(ms.get("sample_rate", 16000))
    n_cfg = int(ms.get("n_mfcc", spec.shape[0]))
    n = min(n_cfg, spec.shape[0])
    spec = spec[:n, :]
    t_grid = np.arange(0, spec.shape[1]) * hop_length / sample_rate
    f_grid = np.arange(spec.shape[0])
    tt, ff = np.meshgrid(t_grid, f_grid)
    f, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(tt, ff, spec, cmap="coolwarm", shading="auto")
    ax.set_xlabel("Time, sec", size=20)
    ax.set_ylabel("MFCC coefficient index", size=20)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = ToTensor()(Image.open(buf)).permute(1, 2, 0).numpy()
    plt.close(f)
    return image


def plot_spectrogram(
    spectrogram: torch.Tensor,
    mel_spec: dict[str, int | float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (26, 7),
) -> np.ndarray:
    """
    Plot a mel-scale spectrogram (log power in dB) and return an RGB array for W&B.

    Args:
        spectrogram: (n_mels, time) or (C, n_mels, time); multi-channel is averaged.
        mel_spec: optional ``hop_length``, ``n_mels``, ``sample_rate`` (torchaudio MelSpectrogram).
        title: optional axis title.

    Returns:
        (H, W, 3) float array in [0, 1].
    """
    spec = spectrogram.detach().float().cpu().numpy()
    if spec.ndim == 3:
        spec = spec.mean(axis=0)
    ms = mel_spec or {}
    hop_length = int(ms.get("hop_length", 160))
    sample_rate = float(ms.get("sample_rate", 16000))
    n_mels_cfg = int(ms.get("n_mels", spec.shape[0]))
    n_mels = min(n_mels_cfg, spec.shape[0])
    spec = spec[:n_mels, :]
    t_grid = np.arange(0, spec.shape[1]) * hop_length / sample_rate
    f_grid = np.arange(spec.shape[0])
    tt, ff = np.meshgrid(t_grid, f_grid)
    f, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        tt,
        ff,
        20 * np.log10(np.maximum(spec, 1e-8)),
        cmap="gist_heat",
        shading="auto",
    )
    ax.set_xlabel("Time, sec", size=20)
    ax.set_ylabel("Frequency, Mel bin", size=20)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = ToTensor()(Image.open(buf)).permute(1, 2, 0).numpy()
    plt.close(f)
    return image


def sphere_plot_tensor(
    embeddings: np.ndarray,
    labels: np.ndarray,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 100,
    title: str | None = None,
) -> torch.Tensor:
    """
    Create a 3D sphere plot of embeddings and return as a torch tensor.

    Expects ``embeddings`` as (N, 3) points on (or near) the unit sphere; rows are
    L2-normalized internally before plotting.

    Args:
        embeddings: (N, 3) array of 3D coordinates.
        labels: (N,) array of labels for coloring.
        figsize: tuple (width, height) in inches.
        dpi: resolution.
        title: optional figure title.

    Returns:
        torch.Tensor of shape (3, H, W) with RGB values in [0, 1].
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    lab = np.asarray(labels)
    if emb.ndim != 2 or emb.shape[1] != 3:
        raise ValueError(f"embeddings must be (N, 3), got {emb.shape}")
    if lab.shape[0] != emb.shape[0]:
        raise ValueError("labels length must match embeddings")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    r = 1.0
    phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color="w", alpha=0.3, linewidth=0
    )

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        emb[:, 2],
        c=lab,
        s=20,
        marker=".",
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    if title:
        ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    image = ToTensor()(Image.open(buf))
    plt.close(fig)
    return image


def embedding_to_3d(emb: torch.Tensor) -> torch.Tensor:
    """Project (N, D) embeddings to (N, 3) via PCA (SVD on centered data)."""
    x = emb.float()
    if x.shape[0] < 2:
        z = torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)
        z[:, : min(3, x.shape[1])] = x[:, : min(3, x.shape[1])]
        return z
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(3, vh.shape[0])
    out = x @ vh[:k].T
    if out.shape[1] < 3:
        pad = torch.zeros(out.shape[0], 3 - out.shape[1], device=out.device, dtype=out.dtype)
        out = torch.cat([out, pad], dim=1)
    return out


def confusion_matrix_figure(
    cm: np.ndarray | torch.Tensor,
    title: str = "confusion_matrix",
    figsize: tuple[float, float] = (8, 7),
):
    """Build a matplotlib figure of a confusion matrix heatmap (returns RGB array for wandb.Image)."""
    if isinstance(cm, torch.Tensor):
        cm = cm.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    return img
