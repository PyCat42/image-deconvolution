import torch
import pandas as pd
from pathlib import Path
import numpy as np

from src.data_helpers import get_dataset_stats
from src.RLdeconvolution import RL_test, RLdeconvolve
from src.ml_helpers import test_model
from src.metric_helpers import psnr, ssim, mae


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

        filepath = dir_path / f"{filename}.csv"
        df.to_csv(filepath, index=False)

    return df

def prior_testing(target, input, psf, model, device="cpu"):
    """
    Test how well NN and RL perform on individual examples.
    Done to determine how strongly NN uses it's prior knowledge.
    :param target: original image
    :param input: blured image
    :param psf: point spread function
    :param model: trained model
    :param device: "cpu" (default) or "cuda
    :return: dataframe containing metric comparison results
    """
    input_tensor = torch.from_numpy(input).float().unsqueeze(0).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)

    # Richardson-Lucy
    rl_estimate = RLdeconvolve(input, psf)
    rl_estimate_tensor = torch.from_numpy(rl_estimate).float().unsqueeze(0).unsqueeze(0)

    # CNN
    model.eval()
    model.to(device)

    with torch.inference_mode():
        cnn_estimate = model(input_tensor)

    cnn_estimate = cnn_estimate.squeeze().cpu().numpy()
    cnn_estimate_tensor = torch.from_numpy(cnn_estimate).float().unsqueeze(0).unsqueeze(0)

    # Metrics
    results = pd.DataFrame({
        "Method": [
            "Blurred",
            "Richardson-Lucy",
            "CNN"
        ],
        "PSNR": [
            psnr(input_tensor, target_tensor).item(),
            psnr(rl_estimate_tensor, target_tensor).item(),
            psnr(cnn_estimate_tensor, target_tensor).item()
        ],
        "SSIM": [
            ssim(input_tensor, target_tensor, device=device).item(),
            ssim(rl_estimate_tensor, target_tensor, device=device).item(),
            ssim(cnn_estimate_tensor, target_tensor, device=device).item()
        ],
        "MAE": [
            mae(input_tensor, target_tensor).item(),
            mae(rl_estimate_tensor, target_tensor).item(),
            mae(cnn_estimate_tensor, target_tensor).item()
        ]
    })

    return results
