import torch
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
from timeit import default_timer

from src.metric_helpers import mae, psnr, ssim, HybridLoss

class EarlyStopping():
    """
    Early stops the training if validation loss doesn't improve by more than min_delta after a given patience.
    """
    def __init__(self, patience=20, min_delta=1e-4, path=None, filename=None):
        self.patience = patience
        self.min_delta = min_delta

        # Track loss improvement
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.epochs_wo_improvement = 0
        self.early_stop = False

        # Where to save best model to (if none, model is not saved)
        self.path = path
        if self.path is not None:
            self.dir_path = Path(path)
            self.dir_path.mkdir(parents=True, exist_ok=True)
        self.filename = filename

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_wo_improvement = 0
            if self.path is not None:
                torch.save(
                    model.state_dict(),
                    self.dir_path / f"{self.filename}.pth"
                )

        else:
            self.epochs_wo_improvement += 1
            if self.epochs_wo_improvement >= self.patience:
                print("***********************************************")
                print(f"No improvement for {self.patience} epochs!")
                self.early_stop = True

def train_model(model, dataloader,
                optimizer, loss_func, device):
    """
    Model training loop.
    :param model:
    :param dataloader:
    :param optimizer:
    :param loss_func:
    :return:
    """
    train_time_start = default_timer()

    # Initialize accumulated losses to 0 at the beginning of each epoch
    accumulated_loss = 0.0
    # ... and MAEs
    accumulated_mae = 0.0
    # ... and PSNR
    accumulated_psnr = 0.0
    # ... and SSIM
    accumulated_ssim = 0.0

    # *** TRAINING ***
    model.train()
    for X, Y in tqdm(dataloader):
        # Move tensors to device
        X, Y = X.to(device), Y.to(device)

        # Forward ppass
        Y_pred = model(X)

        # Calculate the loss and MAE and add to sums
        loss = loss_func(Y_pred, Y)
        accumulated_loss += loss.item()
        accumulated_mae += mae(Y_pred, Y).detach().cpu().item()
        accumulated_psnr += psnr(Y_pred, Y).detach().cpu().item()
        accumulated_ssim += ssim(Y_pred, Y, device=device).detach().cpu().item()

        # Zero-out optimizer gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Step the optimizer
        optimizer.step()

    # Get average loss and MAE per batch
    train_batches = len(dataloader)
    avg_loss = accumulated_loss / train_batches
    avg_mae = accumulated_mae / train_batches
    avg_psnr = accumulated_psnr / train_batches
    avg_ssim = accumulated_ssim / train_batches

    train_time_end = default_timer()

    train_time = train_time_end - train_time_start

    return avg_loss, avg_mae, avg_psnr, avg_ssim, train_time

def validate_model(model, dataloader, loss_func, device):
    """
    Model validation loop.
    :param model:
    :param dataloader:
    :param loss_func:
    :return:
    """
    accumulated_val_loss = 0.0
    accumulated_val_mae = 0.0
    accumulated_val_psnr = 0.0
    accumulated_val_ssim = 0.0

    # *** VALIDATION ***
    model.eval()
    with torch.inference_mode():
        for X_val, Y_val in tqdm(dataloader):
            # Move tensors to device
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            # Make prediction
            val_pred = model(X_val)

            # Calculate the loss and metrics and add to sums
            val_loss = loss_func(val_pred, Y_val)
            accumulated_val_loss += val_loss.item()
            accumulated_val_mae += mae(val_pred, Y_val).detach().cpu().item()
            accumulated_val_psnr += psnr(val_pred, Y_val).detach().cpu().item()
            accumulated_val_ssim += ssim(val_pred, Y_val, device=device).detach().cpu().item()

        # Get average loss and metrics per batch
        validation_batches = len(dataloader)
        avg_val_loss = accumulated_val_loss / validation_batches
        avg_val_mae = accumulated_val_mae / validation_batches
        avg_val_psnr = accumulated_val_psnr / validation_batches
        avg_val_ssim = accumulated_val_ssim / validation_batches

        return avg_val_loss, avg_val_mae, avg_val_psnr, avg_val_ssim

def train_val_loop(model, train_dataloader, validation_dataloader,
                   optimizer, loss_func, scheduler,
                   max_epochs=200, patience=20, min_delta=1e-4, do_early_stopping=True,
                   device='cpu', path=None, filename=None):
    """
    Model training and validation.
    :param model: model to be trained
    :param train_dataloader: dataloader containing training dataset
    :param validation_dataloader: dataloader containing validation dataset
    :param optimizer: optimizer
    :param loss_func: loss function
    :param scheduler: scheduler
    :param max_epochs: maximal number of epochs for which to run training
    :param patience: number of epochs to wait for improvement in loss before invoking early stopping
    :param min_delta: minimal improvement in loss that is considered as such
    :param do_early_stopping: True (default) if early stopping is done
    :param path: path to which models should be saved
    :param filename: name of the file to which to save the models
    :return: train_loss_vals, train_mae_vals, train_psnr_vals, train_ssim_vals, train_time_accumulated,
            validation_loss_vals, validation_mae_vals, validation_psnr_vals, validation_ssim_vals,
            best_epoch, lr_vals
    """
    # Keep track of losses
    train_loss_vals = []
    validation_loss_vals = []
    # ... and MAEs
    train_mae_vals = []
    validation_mae_vals = []
    # ... and PSNR
    train_psnr_vals = []
    validation_psnr_vals = []
    # ... and SSIM
    train_ssim_vals = []
    validation_ssim_vals = []

    # Keep track of learning rate (since there is a scheduler)
    lr_vals = []

    early_stopping = EarlyStopping(
        patience, min_delta,
        path, filename
    )

    train_time_accumulated = 0
    best_epoch = 0
    best_loss = float("inf")

    for epoch in tqdm(range(max_epochs)):
        print(f"\nEPOCH: {epoch}\n----------------------")
        current_lr = optimizer.param_groups[0]['lr']
        lr_vals.append(current_lr)

        avg_loss, avg_mae, avg_psnr, avg_ssim, train_time = train_model(model, train_dataloader, optimizer, loss_func, device=device)
        train_time_accumulated += train_time
        train_loss_vals.append(avg_loss)
        train_mae_vals.append(avg_mae)
        train_psnr_vals.append(avg_psnr)
        train_ssim_vals.append(avg_ssim)

        avg_val_loss, avg_val_mae, avg_val_psnr, avg_val_ssim = validate_model(model, validation_dataloader, loss_func, device=device)
        validation_loss_vals.append(avg_val_loss)
        validation_mae_vals.append(avg_val_mae)
        validation_psnr_vals.append(avg_val_psnr)
        validation_ssim_vals.append(avg_val_ssim)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch

        # Track changes during epochs
        print(f"Train loss: {avg_loss} | Validation loss: {avg_val_loss}")
        print(f"Train MAE: {avg_mae} | Validation MAE: {avg_val_mae}")
        print(f"Train PSNR: {avg_psnr} | Validation PSNR: {avg_val_psnr}")
        print(f"Train SSIM: {avg_ssim} | Validation SSIM: {avg_val_ssim}")
        print(f"Learning rate: {current_lr}")

        # Change learning rate if necessary
        scheduler.step(avg_val_loss)

        # Early stopping
        if do_early_stopping:
            early_stopping(avg_val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f"Stopping early at epoch {epoch + 1}!")
                if path is not None:
                    best_epoch = early_stopping.best_epoch
                    print(f"Best model saved at epoch {best_epoch} to file {filename} in {path}...")
                print("***********************************************")
                break

    return (train_loss_vals, train_mae_vals, train_psnr_vals, train_ssim_vals, train_time_accumulated,
            validation_loss_vals, validation_mae_vals, validation_psnr_vals, validation_ssim_vals,
            best_epoch, lr_vals)

def run_experiment(model, train_dataloader, validation_dataloader, batch_size,
                   mae_share, edge_share, ssim_share, mse_share,
                   learning_rate, scheduler_factor, scheduler_patience,
                   max_epochs, patience, min_delta, do_early_stopping,
                   overlap=0.5,
                   device="cpu", seed=None,
                   path=None, save_name="new_model"):
    """
    Implements raining and validation loop and saving of training stats and trained model parameters.
    :param model: model to be trained
    :param train_dataloader: dataloader containing training dataset
    :param validation_dataloader: dataloader containing validation dataset
    :param batch_size: batch size for training and validation
    :param mae_share: share of MAE loss in hybrid loss
    :param edge_share: share of Edge loss in hybrid loss
    :param ssim_share: share of SSIM loss in hybrid loss
    :param mse_share: share of MSE loss in hybrid loss
    :param learning_rate: initial learning rate
    :param scheduler_factor: factor by which scheduler changes learning rate
    :param scheduler_patience: number of epochs to wait before scheduler changes learning rate
    :param max_epochs: maximal number of epochs
    :param patience: number of epochs to wait for loss improvement before invoking early stopping
    :param min_delta: minimal loss improvement that is considered as such
    :param do_early_stopping: True (default) if early stopping should be executed
    :param overlap: parameter that determines how hard dataset is when artificial dataset is used
    :param device: "cpu" (default) or "cuda
    :param seed: random seed for reproducibility
    :param path: path to which models and data frames are saved
    :param save_name: prefix with which files are to be saved
    :return:
    """

    # Calculate number of trainable parameters
    num_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Define loss function
    loss_func = HybridLoss(
        device,
        mae_share=mae_share,
        edge_share=edge_share,
        ssim_share=ssim_share,
        mse_share=mse_share
    ).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience
    )

    # Training & Validation
    (train_loss_vals, train_mae_vals, train_psnr_vals, train_ssim_vals,
     train_time, validation_loss_vals, validation_mae_vals, validation_psnr_vals, validation_ssim_vals,
     best_epoch, lr_vals) = train_val_loop(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_func=loss_func,
        scheduler=scheduler,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        do_early_stopping=do_early_stopping,
        device=device,
        path=path,
        filename=f"{save_name}.pth"
    )

    # Save metrics values from each epoch to dataframe
    df_metrics = pd.DataFrame({
        "train_loss_vals": train_loss_vals,
        "train_mae_vals": train_mae_vals,
        "train_psnr_vals": train_psnr_vals,
        "train_ssim_vals": train_ssim_vals,

        "validation_loss_vals": validation_loss_vals,
        "validation_mae_vals": validation_mae_vals,
        "validation_psnr_vals": validation_psnr_vals,
        "validation_ssim_vals": validation_ssim_vals,

        "learning_rate_vals": lr_vals
    })

    # Save general model information to dataframe
    df_about = pd.DataFrame({
        "rnd_seed": seed,

        "overlap_share": overlap,

        "num_trainable_params": num_trainable_params,

        "batch_size": batch_size,

        "mae_share": mae_share,
        "edge_share": edge_share,
        "ssim_share": ssim_share,
        "mse_share": mse_share,

        "optimizer": "Adam",
        "initial_learning_rate": learning_rate,

        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,

        "train_time": train_time,

        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "best_epoch_train_loss": train_loss_vals[best_epoch],
        "best_epoch_train_mae": train_mae_vals[best_epoch],
        "best_epoch_train_psnr": train_psnr_vals[best_epoch],
        "best_epoch_train_ssim": train_ssim_vals[best_epoch],
        "best_epoch_validation_loss": validation_loss_vals[best_epoch],
        "best_epoch_validation_mae": validation_mae_vals[best_epoch],
        "best_epoch_validation_psnr": validation_psnr_vals[best_epoch],
        "best_epoch_validation_ssim": validation_ssim_vals[best_epoch]
    })

    # Saving
    if path is not None:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        df_metrics.to_csv(
            dir_path / f"{save_name}_metrics.csv",
            index=False
        )
        df_about.to_csv(
            dir_path / f"{save_name}_about.csv",
            index=False
        )

    return df_metrics, df_about

def test_model(model, model_about_df, test_dataloader, device="cpu"):
    """
    Test loop implementation.
    :param model: trained model
    :param model_about_df: dictionary containing information about the model
    :param test_dataloader: dataloader with test dataset
    :param device: "cpu" (default) or "cuda"
    :return:
    """
    # Recreate same loss used for training
    loss_func = HybridLoss(
        device,
        mae_share=model_about_df["mae_share"].item(),
        edge_share=model_about_df["edge_share"].item(),
        ssim_share=model_about_df["ssim_share"].item(),
        mse_share=model_about_df["mse_share"].item()
    ).to(device)

    # Initialize accumulated loss and metrics to 0
    test_loss_vals = []
    test_mae_vals = []
    test_psnr_vals = []
    test_ssim_vals = []

    # Testing loop
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X_test, Y_test in tqdm(test_dataloader):
            # Move tensors to device
            X_test, Y_test = X_test.to(device), Y_test.to(device)

            # Make prediction
            test_pred = model(X_test)

            # Calculate the loss and add to the sum
            test_loss = loss_func(test_pred, Y_test)
            test_loss_vals.append(test_loss.item())
            test_mae_vals.append(mae(test_pred, Y_test).detach().cpu().item())
            test_psnr_vals.append(psnr(test_pred, Y_test).detach().cpu().item())
            test_ssim_vals.append(ssim(test_pred, Y_test, device=device).detach().cpu().item())

    return test_loss_vals, test_mae_vals, test_psnr_vals, test_ssim_vals
