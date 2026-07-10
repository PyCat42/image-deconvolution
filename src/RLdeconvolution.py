"""
Richardson-Lucy algorithm implementation and related functions.
"""
import numpy as np
import torch
from scipy.ndimage import convolve
from skimage.restoration import denoise_tv_chambolle
from tqdm.auto import tqdm

from src.metric_helpers import mae, psnr, ssim


def RL_step(detected, estimate, psf, eps=1e-12):
    """
    Single step of Richardson-Lucy deconvolution algorithm.
    :param detected: original lower quality image
    :param psf: point spread function
    :param eps: added for numerical stability because algorithm assumes nonnegative images
    :return: enhanced image
    """
    # Step 1: Convolve estimated image with PSF
    # (blur the estimate)
    blurred = convolve(
        estimate,
        psf,
        mode="reflect"
    )

    # Step 2: Divide observed image by convolved image
    # (how similar is this blured estimate to what is detected)
    ratio = detected / np.maximum(blurred, eps)

    correction = convolve(
        ratio,
        psf[::-1, ::-1],
        mode="reflect"
    )

    # Step 3: Update estimated image
    # (update based on similarity observed in previous step)
    estimate = estimate * correction

    estimate = np.maximum(estimate, eps)

    return estimate

def RLdeconvolve(detected, psf, num_iter=None):
    """
    Richardson-Lucy deconvolution algorithm implementation.
    Library implementation:
    skimage.restoration.richardson_lucy(image, psf, num_iter=50, clip=True, filter_epsilon=None)
    (https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.richardson_lucy)
    Implemented here by hand to enable better control.
    :param detected: original lower quality image
    :param psf: point spread function
    :param num_iter: number of algorithm iterations
    :return: enhanced image
    """
    # Image is measured at the detector
    # We then want to improve our estimate when we know PSF
    # Initial estimate is measured image itself
    estimate = detected.astype(np.float32).copy()

    # Flip PSF
    psf = psf.astype(np.float32)
    psf /= psf.sum()

    # Apply iterative RL deconvolution steps:
    for _ in range(num_iter):
        estimate = RL_step(detected, estimate, psf)

    return np.clip(estimate, 0, 1)

def RL_test(dataset, min_delta=1e-4,
            lambda_ssim=5.0, gamma_mae=100,
            regularization_alpha=0.0):
    """
    Runs RL algorithm on a given image dataset.
    Uses oracle score minimization based on knowledge of
    original image (ideal reconstruction) for early stopping.
    :param dataset: test dataset (containing input_img, target, psf)
    :param min_delta: minimal difference in oracle score that is registered as an improvement
    :param lambda_ssim: SSIM coefficient in oracle score (default: 5.0)
    :param gamma_mae: MAE coefficient in oracle score (default: 100)
    :param regularization_alpha: alpha used for TV regularization
                                utilizes denoise_tv_chambolle function
                                (default: 0.0, i.e. no regularization)
    :return: lists iter_vals, initial_psnr_vals, psnr_vals, mae_vals, ssim_vals
            containing these values for each image from the dataset
    """
    # Save metrics
    initial_psnr_vals = []
    psnr_vals = []
    ssim_vals = []
    mae_vals = []
    iter_vals = []

    # We use oracle stopping criterion here
    # In reality we wouldn't have access to ground truth and would have to rely on different early stopping methods
    # There are two main reasons that justify the use of oracle score:
    # - we want to compare absolutely best version of RL algorithm to NN
    # - this approach is feasible option considering we have to evaluate on large dataset
    max_iter = 200
    patience = 10 # if improvement doesn't happen in this many iterations stop algorithm

    # Loop through dataset
    for idx in tqdm(range(len(dataset))):
        input_img, target, psf, *_ = dataset[idx]

        detected = input_img.squeeze().numpy().astype(np.float32)
        target = target.squeeze().numpy().astype(np.float32)

        estimate = detected.copy()

        # Calculate initial metrics
        estimate_tensor = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
        initial_psnr = psnr(estimate_tensor, target_tensor).item()
        initial_ssim = ssim(estimate_tensor, target_tensor).item()
        initial_mae = mae(estimate_tensor, target_tensor).item()

        # Calculate initial oracle score and set it as the best score value
        best_score = initial_psnr + (lambda_ssim * initial_ssim) - (gamma_mae * initial_mae)
        # Keep track of PSNR, SSIM and MAE in the best iteration...
        best_psnr = initial_psnr
        best_ssim = initial_ssim
        best_mae = initial_mae
        # ...as well as the best iteration number...
        best_iter = 0
        # ...and the best estimate
        iters_wo_improvement = 0

        # RL loop
        for i in range(max_iter):
            # do one RL step
            estimate = RL_step(detected, estimate, psf)

            # apply TV regularization
            if regularization_alpha > 0 and i % 2 == 0:
                estimate = denoise_tv_chambolle(estimate, weight=regularization_alpha)

            current_estimate_tensor = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0)
            current_target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)

            # get metric values in this iteration and calculate oracle score
            current_psnr = psnr(current_estimate_tensor, current_target_tensor).item()
            current_ssim = ssim(current_estimate_tensor, current_target_tensor).item()
            current_mae = mae(current_estimate_tensor, current_target_tensor).item()
            current_score = current_psnr + (lambda_ssim * current_ssim) - (gamma_mae * current_mae)

            # check against the best score
            if current_score > best_score + min_delta:
                iters_wo_improvement = 0
                best_psnr = current_psnr
                best_ssim = current_ssim
                best_mae = current_mae
                best_score = current_score
                best_iter = i + 1
            else:
                iters_wo_improvement += 1
                # if score doesn't improve for patience iterations stop
                if iters_wo_improvement >= patience:
                    break

        # save the best values for each image from the dataset
        initial_psnr_vals.append(initial_psnr)
        psnr_vals.append(best_psnr)
        ssim_vals.append(best_ssim)
        mae_vals.append(best_mae)
        iter_vals.append(best_iter)

    return iter_vals, initial_psnr_vals, psnr_vals, mae_vals, ssim_vals

def get_RL_estimate(input_img, target, psf, regularization_alpha=0.0):
    """
    Runs RL algorithm on a given input image.
    Uses oracle score minimization based on knowledge of
    original image (ideal reconstruction) for early stopping.
    :param input_img: blurred degraded image
    :param target: original sharp image
    :param psf: point spread function matrix
    :param regularization_alpha: alpha used for TV regularization
                                utilizes denoise_tv_chambolle function
                                (default: 0.0, i.e. no regularization)
    :return: best_estimate, best_iter, initial_psnr, best_psnr, best_ssim, best_mae
    """
    max_iter = 300
    patience = 10
    min_delta = 1e-4
    lambda_ssim = 5.0
    gamma_mae = 100

    detected = input_img.squeeze().numpy().astype(np.float32)
    target = target.squeeze().numpy().astype(np.float32)

    # Initial estimate is detected image itself
    estimate = detected.copy()

    # Calculate initial metrics
    estimate_tensor = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    initial_psnr = psnr(estimate_tensor, target_tensor).item()
    initial_ssim = ssim(estimate_tensor, target_tensor).item()
    initial_mae = mae(estimate_tensor, target_tensor).item()

    # Set best oracle score, metrics and estimates to initial values
    best_score = initial_psnr + (lambda_ssim * initial_ssim) - (gamma_mae * initial_mae)
    best_psnr = initial_psnr
    best_ssim = initial_ssim
    best_mae = initial_mae
    best_iter = 0
    best_estimate = estimate.copy()
    iters_wo_improvement = 0

    for i in range(max_iter):
        # Do RL step
        estimate = RL_step(detected, estimate, psf)

        # Run regularization
        if regularization_alpha > 0 and i % 2 == 0:
            estimate = denoise_tv_chambolle(estimate, weight=regularization_alpha)

        current_estimate_tensor = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0)
        current_target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)

        # Calculate metrics and score in this iteration
        current_psnr = psnr(current_estimate_tensor, current_target_tensor).item()
        current_ssim = ssim(current_estimate_tensor, current_target_tensor).item()
        current_mae = mae(current_estimate_tensor, current_target_tensor).item()
        current_score = current_psnr + (lambda_ssim * current_ssim) - (gamma_mae * current_mae)

        # Compare against the best value of the score
        if current_score > best_score + min_delta:
            iters_wo_improvement = 0
            best_psnr = current_psnr
            best_ssim = current_ssim
            best_mae = current_mae
            best_score = current_score
            best_iter = i + 1
            best_estimate = estimate.copy()
        else:
            iters_wo_improvement += 1
            # if score doesn't improve for patience iterations stop
            if iters_wo_improvement >= patience:
                break

    return best_estimate, best_iter, initial_psnr, best_psnr, best_ssim, best_mae