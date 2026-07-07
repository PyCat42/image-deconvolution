import torch
from tqdm.auto import tqdm

from metric_helpers import mae, psnr, ssim

class EarlyStopping():
    """
    Early stops the training if validation loss doesn't improve by more than min_delta after a given patience.
    """
    def __init__(self, patience=20, min_delta=1e-4, path=None):
        self.patience = patience
        self.min_delta = min_delta

        # Track loss impovement
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.epochs_wo_improvement = 0
        self.early_stop = False

        # Where to save best model to (if none, model is not saved)
        self.path = path

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_wo_improvement = 0
            if self.path is not None:
                torch.save(model.state_dict(), self.path)
        else:
            self.epochs_wo_improvement += 1
            if self.epochs_wo_improvement >= self.patience:
                print("***********************************************")
                print(f"No improvement for {self.patience} epochs!")
                self.early_stop = True

def train_model(model, train_dataloader,
                optimizer, loss):
    """
    Model training loop.
    :param model:
    :param train_dataloader:
    :param optimizer:
    :param loss:
    :param mae:
    :param psnr:
    :param ssim:
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
    for X, Y, *_ in tqdm(train_dataloader):
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
    avg_loss = accumulated_loss / train_batches
    avg_mae = accumulated_mae / train_batches
    avg_psnr = accumulated_psnr / train_batches
    avg_ssim = accumulated_ssim / train_batches

    train_time_end = default_timer()

    train_time = train_time_end - train_time_start

    return avg_loss, avg_mae, avg_psnr, avg_ssim, train_time

def validate_model(model, train_dataloader, loss):
    """
    Model validation loop.
    :param model:
    :param train_dataloader:
    :param loss:
    :param mae:
    :param psnr:
    :param ssim:
    :return:
    """
    accumulated_val_loss = 0.0
    accumulated_val_mae = 0.0
    accumulated_val_psnr = 0.0
    accumulated_val_ssim = 0.0

    # *** VALIDATION ***
    model.eval()
    with torch.inference_mode():
        for X_val, Y_val, *_ in tqdm(validation_dataloader):
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
        avg_val_loss = accumulated_val_loss / validation_batches
        avg_val_mae = accumulated_val_mae / validation_batches
        avg_val_psnr = accumulated_val_psnr / validation_batches
        avg_val_ssim = accumulated_val_ssim / validation_batches

        return avg_val_loss, avg_val_mae, avg_val_psnr, avg_val_ssim

def train_val_loop(model, train_dataloader,
                  optimizer, loss,
                  scheduler, max_epochs,
                  patience, min_delta, do_early_stopping=True,
                  path=None):
    """
    Model training and validation.
    :param model:
    :param train_dataloader:
    :param optimizer:
    :param loss:
    :param mae:
    :param psnr:
    :param ssim:
    :param scheduler:
    :param max_epochs:
    :param patience:
    :param min_delta:
    :param do_early_stopping:
    :param path:
    :return:
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

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        path=path
    )

    train_time_accumulated = 0

    for epoch in tqdm(range(max_epochs)):
        print(f"\nEPOCH: {epoch}\n----------------------")

        avg_loss, avg_mae, avg_psnr, avg_ssim, train_time = train_model(model, train_dataloader, optimizer, loss)
        train_time_accumulated += train_time
        train_loss_vals.append(avg_loss)
        train_mae_vals.append(avg_mae)
        train_psnr_vals.append(avg_psnr)
        train_ssim_vals.append(avg_ssim)

        avg_val_loss, avg_val_mae, avg_val_psnr, avg_val_ssim = validate_model(model, train_dataloader, loss)
        validation_loss_vals.append(avg_val_loss)
        validation_mae_vals.append(avg_val_mae)
        validation_psnr_vals.append(avg_val_psnr)
        validation_ssim_vals.append(avg_val_ssim)

        # Track changes during epochs
        print(f"Train loss: {avg_loss} | Validation loss: {avg_val_loss}")
        print(f"Train MAE: {avg_mae} | Validation MAE: {avg_val_mae}")
        print(f"Train PSNR: {avg_psnr} | Validation PSNR: {avg_val_psnr}")
        print(f"Train SSIM: {avg_ssim} | Validation SSIM: {avg_val_ssim}")

        # Change learning rate if necessary
        scheduler.step(avg_val_loss)

        # Early stopping
        if do_early_stopping:
            early_stopping(avg_val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f"Stopping early at epoch {epoch + 1}!")
                if path is not None:
                    print(f"Best model saved at epoch {early_stopping.best_epoch} to {path}...")
                print("***********************************************")
                break

    return (train_loss_vals, train_mae_vals, train_psnr_vals, train_ssim_vals, train_time_accumulated,
            validation_loss_vals, validation_mae_vals, validation_psnr_vals, validation_ssim_vals,
            early_stopping.best_epoch)
