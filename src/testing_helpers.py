import os.path

import torch
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data_helpers import get_dataset_stats
from src.RLdeconvolution import RL_test, RLdeconvolve, get_RL_estimate
from src.ml_helpers import test_model
from src.metric_helpers import psnr, ssim, mae
from src.plotting_helpers import plot_reconstruction_comparison


def model_comparison_table(model_files, filename=None):
    """
    Compiles number of params, training times, the best epoch numbers
     and best validation metrics across all models.
    :param model_files: dictionary of model name -> file path
    :param filename: name of the file to which to save the comparison table
    :return:
    """
    summary = []
    columns = [
        "num_trainable_params",
        "train_time",
        "best_epoch",
        "best_epoch_validation_loss",
        "best_epoch_validation_mae",
        "best_epoch_validation_psnr",
        "best_epoch_validation_ssim"
    ]
    names = [
        "Params Num",
        "Train Time (s)",
        "Best Epoch",
        "Validation Loss",
        "Validation MAE",
        "Validation PSNR",
        "Validation SSIM"
    ]

    for model_name, file_path in model_files.items():
        path = Path(file_path)
        df = pd.read_csv(path)

        new_row = {"Model Name": model_name}
        for col, name in zip(columns, names):
            new_row[name] = df[col].iloc[0]

        summary.append(new_row)

    df_summary = pd.DataFrame(summary)

    if filename is not None:
        df_summary.to_csv(filename)

    return df_summary

def comparative_testing(dataset, dataloader, model, model_about_df,
                        device="cpu", path="comparisson", filename="comp_df.csv"):
    """
    Runs Richardson-Lucy and NN model on same dataset.
    :param dataset: test dataset
    :param dataloader: dataloader containing the same test dataset
    :param model: trained model
    :param model_about_df: dictionary containing model metadata (created by run_experiment function)
    :param device: "cpu" (default) or "cuda"
    :param path: path to which to save outputs
    :param filename: prefix with which to save outputs
    :return: data frame containing comparison results
            (rl and nn metric values for each image in dataset
            and characteristics related to blurring of each image)
    """
    # Fetch dataset characteristics
    sigma_x_vals, sigma_y_vals, kernel_size_vals, photon_flux_vals = get_dataset_stats(dataset)

    # Run Richardson-Lucy algorithm on dataset
    iter_vals, initial_psnr_vals, rl_psnr_vals, rl_mae_vals, rl_ssim_vals = RL_test(dataset)
    initial_psnr_vals = np.array(initial_psnr_vals)
    rl_psnr_vals = np.array(rl_psnr_vals)
    rl_delta_psnr_vals = rl_psnr_vals - initial_psnr_vals

    # Run NN model on dataset
    nn_loss_vals, nn_mae_vals, nn_psnr_vals, nn_ssim_vals = test_model(model, model_about_df, dataloader, device)
    nn_psnr_vals = np.array(nn_psnr_vals)
    nn_delta_psnr_vals = nn_psnr_vals - initial_psnr_vals

    df= pd.DataFrame({
        "sigma_x": sigma_x_vals,
        "sigma_y": sigma_y_vals,
        "kernel_size": kernel_size_vals,
        "photon_flux": photon_flux_vals,
        "iterations": iter_vals,
        "initial_psnr": initial_psnr_vals,

        "rl_psnr_vals": rl_psnr_vals,
        "rl_delta_psnr_vals": rl_delta_psnr_vals,
        "rl_mae_vals": rl_mae_vals,
        "rl_ssim_vals": rl_ssim_vals,

        "nn_loss_vals": nn_loss_vals,
        "nn_psnr_vals": nn_psnr_vals,
        "nn_delta_psnr_vals": nn_delta_psnr_vals,
        "nn_mae_vals": nn_mae_vals,
        "nn_ssim_vals": nn_ssim_vals,
    })

    if path is not None:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        filepath = dir_path / filename
        df.to_csv(os.path.join(dir_path, filepath), index=False)

    return df

def prior_testing(target, input_img, psf, model,
                  max_iter=300, patience = 10, min_delta = 1e-4,
                  lambda_ssim = 5.0, gamma_mae = 100,
                  regularization_alpha=0.0,
                  device="cpu", path="prior_testing", save_name="prior_test"):
    """
    Test how well NN and RL perform on individual examples.
    Done to determine how strongly NN uses it's prior knowledge.
    :param target: original image
    :param input_img: blured image
    :param psf: point spread function
    :param model: trained model
    :param max_iter: maximum number of RL iterations
    :param patience: number of RL iterations without improvement before early stopping is invoked
    :param min_delta: minimal oracle score improvement that is registered as such
    :param lambda_ssim: SSIM coefficient in oracle score
    :param gamma_mae: MAE coefficient in oracle score
    :param regularization_alpha: alpha used for TV regularization
                                utilizes denoise_tv_chambolle function
                                (default: 0.0, i.e. no regularization)
    :param device: "cpu" (default) or "cuda"
    :param path: path to which to save outputs
    :param save_name: name of the file to which to save created plots
    :return: dataframe containing metric comparison results
            plots this dataframe as table
            creates side by side plot of degraded, RL reconstructed, NN reconstructed and original image
    """
    if path is not None:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = torch.from_numpy(input_img).float().unsqueeze(0).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)

    # Richardson-Lucy
    rl_estimate, rl_best_iter, _, rl_best_psnr, rl_best_ssim, rl_best_mae = get_RL_estimate(
        detected=input_img,
        target=target,
        psf=psf,
        max_iter=max_iter,
        patience=patience,
        min_delta=min_delta,
        lambda_ssim=lambda_ssim,
        gamma_mae=gamma_mae,
        regularization_alpha=regularization_alpha)

    # CNN
    model.eval()
    model.to(device)

    with torch.inference_mode():
        cnn_estimate = model(input_tensor)

    cnn_estimate = cnn_estimate.squeeze().cpu().numpy()
    cnn_estimate_tensor = torch.from_numpy(cnn_estimate).float().unsqueeze(0).unsqueeze(0)

    # Blurred metrics
    blurred_psnr = psnr(input_tensor, target_tensor).item()
    blurred_ssim = ssim(input_tensor, target_tensor, device=device).item()
    blurred_mae = mae(input_tensor, target_tensor).item()

    # CNN metrics
    cnn_psnr = psnr(cnn_estimate_tensor, target_tensor).item()
    cnn_ssim = ssim(cnn_estimate_tensor, target_tensor, device=device).item()
    cnn_mae = mae(cnn_estimate_tensor, target_tensor).item()

    results = pd.DataFrame({
        "Method": [
            "Blurred",
            "Richardson-Lucy",
            "CNN"
        ],
        "PSNR": [
            blurred_psnr,
            rl_best_psnr,
            cnn_psnr
        ],
        "SSIM": [
            blurred_ssim,
            rl_best_ssim,
            cnn_ssim
        ],
        "MAE": [
            blurred_mae,
            rl_best_mae,
            cnn_mae
        ],
        "PSNR vs Blurred": [
            0.0,
            f"{100 * (rl_best_psnr - blurred_psnr) / blurred_psnr:.2f} %",
            f"{100 * (cnn_psnr - blurred_psnr) / blurred_psnr:.2f} %"
        ],
        "SSIM vs Blurred": [
            0.0,
            f"{100 * (rl_best_ssim - blurred_ssim) / blurred_ssim:.2f} %",
            f"{100 * (cnn_ssim - blurred_ssim) / blurred_ssim:.2f} %"
        ],
        "MAE vs Blurred": [
            0.0,
            f"{100 * (blurred_mae - rl_best_mae) / blurred_mae:.2f} %",
            f"{100 * (blurred_mae - cnn_mae) / blurred_mae:.2f} %"
        ]
    })

    results.to_csv(os.path.join(save_dir, f"{save_name}_table.csv"), index=False)

    # Plot created dataframe as table
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=results.round(4).values,
        colLabels=results.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Colors (25% saturation = 75% white mixed)
    blurred_color = "#DDDDDD"   # gray
    rl_color = "#BFEFFF"        # light cyan (25% saturation)
    nn_color = "#FFBFFF"        # light magenta (25% saturation)

    row_colors = {
        "Blurred": blurred_color,
        "Richardson-Lucy": rl_color,
        "CNN": nn_color
    }

    # Apply row colors
    for i, method in enumerate(results["Method"]):
        for j in range(len(results.columns)):
            table[(i + 1, j)].set_facecolor(row_colors[method])

    # Header styling
    for j in range(len(results.columns)):
        table[(0, j)].set_text_props(weight="bold")

    plt.tight_layout()

    if save_dir is not None and save_name is not None:
        plt.savefig(os.path.join(save_dir, f"{save_name}_table.png"), dpi=300)

    plt.show()

    plot_reconstruction_comparison(
        target=target,
        input_img=input_img,
        rl_estimate=rl_estimate,
        rl_best_iter=rl_best_iter,
        cnn_estimate=cnn_estimate,
        path=save_dir,
        save_name=save_name
    )

    return results
