import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import lab2rgb


def lab_to_rgb(L, ab):
    """
    Convert LAB tensor batch to numpy RGB images.

    Parameters
    ----------
    L  : [B, 1, H, W] tensor, normalized to [-1, 1]
    ab : [B, 2, H, W] tensor, normalized to [-1, 1]

    Returns
    -------
    np.ndarray [B, H, W, 3], values in [0, 1]
    """
    L  = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    return np.stack([lab2rgb(img) for img in Lab], axis=0)


def visualize(model, data, n=5, save=False, save_path="result.png"):
    """
    Show a grid of: grayscale input / colorized output / ground truth.

    Parameters
    ----------
    model : MainModel
    data  : batch dict with 'L' and 'ab'
    n     : number of images to show (up to batch size)
    """
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()

    n = min(n, len(model.L))
    fake = lab_to_rgb(model.L[:n], model.fake_color[:n].detach())
    real = lab_to_rgb(model.L[:n], model.ab[:n])

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i in range(n):
        axes[0, i].imshow(model.L[i][0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(fake[i])
        axes[1, i].axis('off')
        axes[2, i].imshow(real[i])
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel("input",        fontsize=11)
    axes[1, 0].set_ylabel("colorized",    fontsize=11)
    axes[2, 0].set_ylabel("ground truth", fontsize=11)

    plt.tight_layout()
    if save:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()


def visualize_uncertainty(model, data, n=5, save=False, save_path="uncertainty.png"):
    """
    Show a grid of: grayscale / colorized / ground truth / uncertainty heatmap.

    Only works when model.use_uncertainty = True.
    The uncertainty heatmap shows per-pixel color uncertainty: bright = unsure,
    dark = confident.

    Parameters
    ----------
    model : MainModel with use_uncertainty=True
    data  : batch dict with 'L' and 'ab'
    """
    assert model.use_uncertainty, "model.use_uncertainty must be True"

    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()

    n = min(n, len(model.L))
    fake        = lab_to_rgb(model.L[:n], model.fake_color[:n].detach())
    real        = lab_to_rgb(model.L[:n], model.ab[:n])
    uncertainty = model.uncertainty[:n, 0].cpu().numpy()  # [n, H, W]

    # normalize uncertainty to [0, 1] for display
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if u_max > u_min:
        uncertainty_norm = (uncertainty - u_min) / (u_max - u_min)
    else:
        uncertainty_norm = uncertainty

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12))
    row_labels = ["input", "colorized", "ground truth", "uncertainty"]

    for i in range(n):
        axes[0, i].imshow(model.L[i][0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(fake[i])
        axes[1, i].axis('off')
        axes[2, i].imshow(real[i])
        axes[2, i].axis('off')
        im = axes[3, i].imshow(uncertainty_norm[i], cmap='hot', vmin=0, vmax=1)
        axes[3, i].axis('off')

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11)

    # colorbar for uncertainty row
    fig.colorbar(im, ax=axes[3, :].tolist(), fraction=0.015, pad=0.01,
                 label="uncertainty (normalized)")

    plt.suptitle("Colorization Uncertainty Map", fontsize=13, y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()


def plot_training_curves(history, save=False, save_path="training_curves.png"):
    """
    Plot loss and metric curves from a TrainingHistory object.

    Parameters
    ----------
    history : TrainingHistory
        Populated by calling history.record(...) at the end of each epoch.
    """
    records = history.records
    epochs  = list(range(1, len(next(iter(records.values()))) + 1))

    # figure out which keys to group
    loss_keys   = [k for k in records if k.startswith('loss_')]
    metric_keys = [k for k in records if k in ('psnr', 'ssim')]
    extra_keys  = [k for k in records if k not in loss_keys and k not in metric_keys and k != 'epoch']

    n_panels = (1 if loss_keys else 0) + len(metric_keys) + (1 if extra_keys else 0)
    if n_panels == 0:
        print("nothing to plot")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    if loss_keys:
        ax = axes[ax_idx]; ax_idx += 1
        for k in loss_keys:
            ax.plot(epochs, records[k], label=k, linewidth=1.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title("Training losses")
        ax.legend(fontsize=7)

    for k in metric_keys:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, records[k], 'o-', linewidth=1.5, markersize=4)
        ax.set_xlabel("epoch"); ax.set_ylabel(k.upper())
        ax.set_title(f"Validation {k.upper()}")

    if extra_keys:
        ax = axes[ax_idx]
        for k in extra_keys:
            ax.plot(epochs, records[k], label=k, linewidth=1.5)
        ax.set_xlabel("epoch"); ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()


def plot_uncertainty_histogram(model, data_loader, n_batches=20, save=False,
                               save_path="uncertainty_hist.png"):
    """
    Plot the distribution of per-pixel uncertainty values across the validation set.

    Useful to confirm that the model is not trivially always uncertain or always
    confident, and that uncertainty correlates with ambiguous regions.
    """
    assert model.use_uncertainty

    model.net_G.eval()
    all_uncertainty = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= n_batches:
                break
            model.setup_input(data)
            model.forward()
            all_uncertainty.append(model.uncertainty.cpu().numpy().ravel())

    model.net_G.train()

    all_uncertainty = np.concatenate(all_uncertainty)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_uncertainty, bins=80, density=True, color='steelblue', alpha=0.8)
    ax.set_xlabel("per-pixel uncertainty (std of predicted ab distribution)")
    ax.set_ylabel("density")
    ax.set_title("Distribution of colorization uncertainty")

    plt.tight_layout()
    if save:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
