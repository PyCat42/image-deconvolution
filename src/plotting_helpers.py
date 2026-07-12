import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import torch
from pathlib import Path
import os


def side_by_side_comp(clean, blurred, estimate=None, filename=None):
    """
    Plots detected image and estimated original side by side.
    :param image: detected image
    :param estimate: estimated original
    :param filename: name of the file to which output figure is saved
    :return:
    """
    if estimate is not None:
        img_num = 3
    else:
        img_num = 2

    fig, ax = plt.subplots(1, img_num, figsize=(11, 5))

    ax[0].imshow(clean, cmap='gray', vmin=0.0, vmax=1.0)
    ax[0].set_title('Original')

    ax[1].imshow(blurred, cmap='gray', vmin=0.0, vmax=1.0)
    ax[1].set_title('Detected')

    if estimate is not None:
        ax[2].imshow(estimate, cmap='gray', vmin=0.0, vmax=1.0)
        ax[2].set_title('Estimate')

    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show()

def plot_random_examples(dataset, model=None,
                         num_examples=3,
                         device="cpu", seed=None):
    """
    Plots original, blurred and reconstructed image for random dataset examples.
    :param dataset: test dataset
    :param model: trained NN model
    :param num_examples: number of examples to plot
    :param device: "cpu" (default) or "cuda"
    :param seed: random seed for reproducibility
    :return:
    """
    if seed:
        random.seed(seed)

    # Select random examples
    total_samples = len(dataset)
    random_idx = random.sample(range(total_samples), min(num_examples, total_samples))

    # Run evaluation
    for count, idx in enumerate(random_idx):
        x, y = dataset[idx]
        clean_img_np = y.squeeze(0).detach().cpu().numpy()
        blurred_img = x

        if model is not None:
            model.to(device)
            model.eval()
            with torch.inference_mode():
                estimated_img = model(blurred_img.unsqueeze(0).to(device))
            estimated_img_np = estimated_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        else:
            estimated_img_np = None

        blurred_img_np = blurred_img.squeeze(0).detach().cpu().numpy()

        # Plot
        side_by_side_comp(clean=clean_img_np, blurred=blurred_img_np, estimate=estimated_img_np)

def plot_curves(epochs_range, train_vals, validation_vals, var_to_plot="Loss", filename=None):
    """
    Plots training and validation curves for various variables.
    :param epochs_range: list of epoch values
    :param train_vals: list of training values or None
    :param validation_vals: list of validation values or None
    :param var_to_plot: "Loss", "Accuracy", "MAE", "PSNR", "SSIM"
    :param filename: if specified, name of the file to which figure will be saved (None by default)
    :return:
    """
    if train_vals is not None:
        plt.plot(epochs_range, train_vals, label="Train")
    if validation_vals is not None:
        plt.plot(epochs_range, validation_vals, label="Validation")
    plt.title(f"{var_to_plot} Curves")
    plt.xlabel("Epochs")
    plt.ylabel(f"Average {var_to_plot} per Batch")
    plt.legend()

    if filename:
        plt.savefig(filename)

    plt.show()

def validation_metrics_comparison(model_files, metric_name="validation_loss_vals", filename=None):
    """
    Plots chosen validation metric for multiple models on the same plot for comparison.
    :param model_files:
    :param metric_name:
    :param filename:
    :return:
    """
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("cool")
    num_models = len(model_files)

    for idx, (model_name, file_path) in enumerate(model_files.items()):
        # Load model dataframe
        path = Path(file_path)
        df = pd.read_csv(path)
        if metric_name not in df.columns:
            print(f"{metric_name} column not present in dataframe")
            continue

        # Pick color for a model
        color_idx = idx / (num_models - 1)
        color = cmap(color_idx)

        epochs = range(1, len(df) + 1)
        plt.plot(epochs, df[metric_name], label=model_name, color=color, linewidth=2)

    # Formatting the plot
    clean_title = metric_name.replace("_vals", "").replace("_", " ")
    plt.title(f"Model Comparison: {clean_title}")
    plt.xlabel("Epochs")
    plt.ylabel(clean_title)

    plt.legend()

    if filename is not None:
        plt.savefig(filename)
    plt.show()

def plot_comparative_results(df, path="comparison", save_name=None):
    """
    Plots comparison between RL and NN results
    :param df: dataframe containing results (created by comparative_testing function)
    :param save_name: prefix with which to save plots
    :return:
    """
    if path is not None:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [
        ("MAE", "rl_mae_vals", "nn_mae_vals"),
        ("PSNR", "rl_psnr_vals", "nn_psnr_vals"),
        ("dPSNR", "rl_delta_psnr_vals", "nn_delta_psnr_vals"),
        ("SSIM", "rl_ssim_vals", "nn_ssim_vals")
    ]

    palette = {"Richardson-Lucy": "cyan", "CNN": "magenta"}

    for metric_name, rl_vals, nn_vals in metrics_to_plot:
        rl_mean, rl_std = df[rl_vals].mean(), df[rl_vals].std()
        nn_mean, nn_std = df[nn_vals].mean(), df[nn_vals].std()

        df_long = pd.melt(
            df[[rl_vals, nn_vals]],
            var_name="Method",
            value_name=metric_name,
        )
        df_long["Method"] = df_long["Method"].map(
            {rl_vals: "Richardson-Lucy", nn_vals: "CNN"}
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), width_ratios=[1.2, 0.8])

        # LEFT: Distribution + Mean/Std
        sns.histplot(
            data=df_long,
            x=metric_name,
            hue="Method",
            kde=True,
            palette=palette,
            ax=axes[0],
            alpha=0.5,
            element="step"
        )

        axes[0].axvline(
            rl_mean,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"RL Mean: {rl_mean:.3f}",
        )
        axes[0].axvline(
            nn_mean,
            color="purple",
            linestyle="--",
            linewidth=2,
            label=f"CNN Mean: {nn_mean:.3f}",
        )

        axes[0].set_title(f"Distribution Comparisson: {metric_name}")
        axes[0].set_xlabel(metric_name)
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # RIGHT: Boxplot
        sns.boxplot(
            data=df_long,
            x="Method",
            y=metric_name,
            hue="Method",
            palette=palette,
            ax=axes[1],
            width=0.5,
            showmeans=True,
            legend=False,
        )
        axes[1].set_title(f"Spread & Outliers: {metric_name}")
        axes[1].set_xlabel("Method")
        axes[1].set_ylabel(metric_name)

        stats_text = (
            "Stats Summary:\n"
            f"RL: {rl_mean:.3f} +- {rl_std:.3f}\n"
            f"CNN: {nn_mean:.3f} +- {nn_std:.3f}"
        )
        axes[1].text(
            0.35,
            0.95,
            stats_text,
            fontsize=8,
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, edgecolor="black"),
        )

        plt.tight_layout()

        if path is not None and save_name is not None:
            plt.savefig(os.path.join(dir_path, f"{save_name}_{metric_name}.png"))
        plt.show()

def plot_reconstruction_comparison(target, input_img,
                                   rl_estimate, rl_best_iter,
                                   cnn_estimate,
                                   path=None, save_name="reco_comp"):
    """
    Creates side by side plot of degraded, RL reconstructed, NN reconstructed and original image
    :param target: original image
    :param input_img: blured image
    :param rl_estimate: Richardson-Lucy estimate of the original image
    :param rl_best_iter: best RL iteration
    :param cnn_estimate: cnn model estimate of the original image
    :param path: path to which to save outputs
    :param save_name: name of the file to which to save created plots
    :return:
    """
    # Plot degraded, RL reconstructed, NN reconstructed and original image side-by-side
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))

    ax[0].imshow(input_img, cmap='gray', vmin=0.0, vmax=1.0)
    ax[0].set_title('Blurred')

    ax[1].imshow(rl_estimate, cmap='gray', vmin=0.0, vmax=1.0)
    ax[1].set_title(f'RL (Best Iter={rl_best_iter})')

    ax[2].imshow(cnn_estimate, cmap='gray', vmin=0.0, vmax=1.0)
    ax[2].set_title('NN')

    ax[3].imshow(target, cmap='gray', vmin=0.0, vmax=1.0)
    ax[3].set_title('Original')

    if save_name is not None:
        if path is not None:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"{save_name}_plot.png"
        else:
            filename = Path(f"{save_name}_plot.png")

        plt.savefig(filename, dpi=300)

    plt.show()