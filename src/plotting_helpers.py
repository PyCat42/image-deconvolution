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

def plot_random_examples(dataset, model=None, num_examples=3, device="cpu", seed=None):
    if seed:
        random.seed(seed)

    total_samples = len(dataset)
    random_idx = random.sample(range(total_samples), min(num_examples, total_samples))

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
