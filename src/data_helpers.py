import numpy as np
import cv2
from scipy import signal
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def apply_degradation(target, size=256,
                      sigma=None, photon_flux=None, dark_map=None,
                      poisson_noise=False, full_detector_model=False,
                      data_loading=True):
    # target as np
    img_np = target.squeeze(0).numpy()

    # Apply blur
    # rather than keeping blur kernel fixed sample it uniformly
    # + different amount of blur along x- and y-axis
    if sigma is None:
        sigma_x = np.random.uniform(1.0, 10.0)
        sigma_y = np.random.uniform(1.0, 10.0)
        sigma_max = max(sigma_x, sigma_y)
    else:
        sigma_x = sigma
        sigma_y = sigma
        sigma_max = sigma

    blur_kernel_size = int(np.ceil(6 * sigma_max))  # so we don't get truncated gaussian
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1  # blur kernel needs to be odd
    blurred = cv2.GaussianBlur(img_np, (blur_kernel_size, blur_kernel_size), sigmaX=sigma_x, sigmaY=sigma_y)
    input_blur = torch.from_numpy(blurred).unsqueeze(dim=0)
    input = torch.clamp(input_blur, 0.0, 1.0)

    # get PSF (needed as input for RL algorithm)
    g_x = cv2.getGaussianKernel(blur_kernel_size, sigma_x)
    g_y = cv2.getGaussianKernel(blur_kernel_size, sigma_y)
    psf = g_y @ g_x.T
    psf = psf / psf.sum()

    if poisson_noise or full_detector_model:
        # Inject additional noise
        if photon_flux is None:
            photon_flux = np.random.uniform(20.0, 100.0)
        else:
            photon_flux = photon_flux
        counts = torch.poisson(input * photon_flux)
        input = counts.float() / photon_flux

    if full_detector_model:
        # Model dark counts - random, per pixel, per frame
        dark_counts = torch.full_like(input, np.random.uniform(0.01, 0.1))

        # Background - external light source
        background = np.random.uniform(0, 0.05)

        # Hot pixels
        hot_pixels = torch.zeros_like(input)
        num_hot_pixels = np.random.randint(10, 50)
        x = np.random.randint(0, size, size=num_hot_pixels)
        y = np.random.randint(0, size, size=num_hot_pixels)

        vals = torch.from_numpy(np.random.uniform(0.2, 0.6, size=num_hot_pixels).astype(np.float32))

        hot_pixels[0, y, x] = vals

        # Create final input
        input = torch.clamp(
            input + dark_counts + hot_pixels + dark_map + background,
            min=0, max=1
        )

    if data_loading:
        return input, target
    else:
        return input, target, psf, sigma_x, sigma_y, blur_kernel_size, photon_flux

class GeometricDataGenerator(Dataset):
    """
    Creates artificial geometric dataset for NN training.
    """
    def __init__(self, data_loading=True, num_samples=1000, size=256,
                 overlap_set_share=0.5,
                 sigma=None, photon_flux=None,
                 poisson_noise=False, full_detector_model=False,
                 seed=None):
        self.data_loading = data_loading  # is this dataset used for NN dataloaders (default: True)
        self.num_samples = num_samples  # number of samples in dataset
        self.size = size  # image size

        self.sigma = sigma  # blur Gaussian sigma (default: None)
        self.photon_flux = photon_flux  # photon flux at detector (default: None)

        self.overlap_set_share = overlap_set_share  # how much do shapes overlap (larger share = harder dataset)

        # add degradation? (default: False)
        self.poisson_noise = poisson_noise
        self.full_detector_model = full_detector_model  # hot pixels, dark counts,...

        self.seed = seed # random seed - need to be the same when training different model versions!!!

        self.dark_map = torch.rand(1, self.size, self.size) * 0.02  # fixed pattern noise - some pixels ALWAYS count larger or smaller values

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Seed the process if seed is defined
        if self.seed is not None:
          unique_seed = self.seed + index
          np.random.seed(unique_seed)
          torch.manual_seed(unique_seed)
          torch.cuda.manual_seed(unique_seed)

        # Creating black canvas
        img = np.zeros((self.size, self.size), dtype=np.float32)

        # Track which areas are occupied by some shape
        occupancy = np.zeros((self.size, self.size), dtype=np.float32)
        candidate_pixels = np.empty((0, 2), dtype=np.int32)

        # Drawing random geometric shapes
        num_shapes = np.random.randint(2, 6)

        for _ in range(num_shapes):
            shape_type = np.random.choice(['line_grid', 'circle', 'rectangle'])
            color = np.random.choice([0.2, 0.4, 0.6, 0.8, 1.0])

            if shape_type == 'line_grid':
                if len(candidate_pixels) > 0 and np.random.rand() < self.overlap_set_share:
                    y, start_x = candidate_pixels[np.random.randint(len(candidate_pixels))]
                else:
                    start_x = np.random.randint(10, self.size - 10)
                line_length = np.random.randint(100, 200)
                spacing = np.random.randint(15, 30)
                thickness = np.random.randint(10, 20)
                num_lines = np.random.randint(1, 5)
                for i in range(num_lines):
                    cv2.line(img, (start_x + i * spacing, 0), (start_x + i * spacing, line_length), color, thickness)

            elif shape_type == 'circle':
                if len(candidate_pixels) > 0 and np.random.rand() < self.overlap_set_share:
                    y, x = candidate_pixels[np.random.randint(len(candidate_pixels))]
                    center = (x, y)
                else:
                    center = (np.random.randint(100, self.size - 100), np.random.randint(100, self.size - 100))
                radius = np.random.randint(100, 150)
                filled = int(np.random.choice([-1, 15]))
                cv2.circle(img, center, radius, color, filled)

            elif shape_type == 'rectangle':
                if len(candidate_pixels) > 0 and np.random.rand() < self.overlap_set_share:
                    y, x = candidate_pixels[np.random.randint(len(candidate_pixels)) ]
                    p1 = (x, y)
                    p2 = (x + np.random.randint(100, 200), y + np.random.randint(100, 200))
                else:
                    p1 = (np.random.randint(100, self.size - 100), np.random.randint(100, self.size - 100))
                    p2 = (p1[0] + np.random.randint(100, 200), p1[1] + np.random.randint(100, 200))
                filled = int(np.random.choice([-1, 15]))
                cv2.rectangle(img, p1, p2, color, filled)

            # Update occupied pixels
            occupancy = np.maximum(occupancy, img)

            # Expand this region by some margin
            kernel = np.ones((15,15), np.uint8)
            candidate_region = cv2.dilate(occupancy, kernel)

            # sample from these pixels to have more full and partial overlaps in the dataset
            candidate_pixels = np.argwhere(candidate_region > 0)

        # Clean image - ground truth
        target = torch.from_numpy(img).unsqueeze(dim=0)

        return apply_degradation(target=target, size=self.size,
                                 sigma=self.sigma, photon_flux=self.photon_flux, dark_map=self.dark_map,
                                 poisson_noise=self.poisson_noise, full_detector_model=self.full_detector_model,
                                 data_loading=self.data_loading)

def rectangle_image(size=256):
    img = np.zeros((size, size), np.float32)

    cv2.rectangle(
        img,
        pt1=(70, 70),
        pt2=(190, 190),
        color=1.0,
        thickness=-1
    )

    return img

def circle_image(size=256):
    img = np.zeros((size, size), np.float32)

    cv2.circle(
        img,
        center=(128, 128),
        radius=60,
        color=1.0,
        thickness=-1
    )

    return img

def line_grid_image(size=256):
    img = np.zeros((size, size), np.float32)

    for x in range(50, 210, 25):
        cv2.line(
            img,
            pt1=(x, 20),
            pt2=(x, 220),
            color=1.0,
            thickness=10
        )

    return img

def blurring_func(img, sigma_x=3.0, sigma_y=3.0, photon_flux=50):
    kernel_size = int(np.ceil(6 * max(sigma_x, sigma_y)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    g_x = cv2.getGaussianKernel(kernel_size, sigma_x)
    g_y = cv2.getGaussianKernel(kernel_size, sigma_y)

    psf = g_y @ g_x.T
    psf /= psf.sum()

    blurred = signal.fftconvolve(img, psf, mode="same")
    blurred[blurred < 0] = 0

    counts = np.random.poisson(blurred * photon_flux)
    input_img = counts / photon_flux

    return input_img, psf


def white_noise_image(size=256, mean=0.5, std=0.15):
    """
    Generates pure uncorrelated Gaussian noise, clipped to [0, 1].
    Provides no structural cues or continuous edges.
    """
    img = np.random.normal(loc=mean, scale=std, size=(size, size)).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

def structured_texture_image(size=256, low_res_size=16):
    """
    Generates a continuous random texture (Speckle/Perlin-like) by
    upscaling a tiny random grid. This creates fine-grained variations
    without any geometric shapes (rectangles, circles, or straight lines).
    """
    # Create a small grid of random values
    small_random = np.random.rand(low_res_size, low_res_size).astype(np.float32)

    # Upscale using bicubic interpolation to create smooth, random texture fields
    img = cv2.resize(small_random, (size, size), interpolation=cv2.INTER_CUBIC)

    # Normalize smoothly between 0 and 1
    img = (img - img.min()) / (img.max() - img.min())
    return img

def get_dataset_stats(dataset):
    """
    Returns statistics about the artificial geometric dataset.
    :param dataset: artificial geometric dataset
    :return: sigma_x_vals, sigma_y_vals, kernel_size_vals, photon_flux_vals
    """
    sigma_x_vals = []
    sigma_y_vals = []
    kernel_size_vals = []
    photon_flux_vals = []
    for idx in range(len(dataset)):
        *_, sigma_x, sigma_y, blur_kernel_size, photon_flux = dataset[idx]

        sigma_x_vals.append(sigma_x)
        sigma_y_vals.append(sigma_y)
        kernel_size_vals.append(blur_kernel_size)
        photon_flux_vals.append(photon_flux)

    return sigma_x_vals, sigma_y_vals, kernel_size_vals, photon_flux_vals

class BBBCDataset(Dataset):
    def __init__(self, image_dir, size=256, sigma=None,
                 poisson_noise=False, full_detector_model=None,
                 data_loading=False, seed=42):
        self.image_dir = image_dir

        self.size = size

        self.sigma = sigma
        self.dark_map = torch.rand(1, self.size, self.size) * 0.02

        self.poisson_noise = poisson_noise
        self.full_detector_model = full_detector_model

        self.data_loading = data_loading

        self.seed = seed

        self.images = sorted(
            os.path.join(image_dir, f) for f in os.listdir(self.image_dir)
            if f.endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png"))
        )

        # Transform to adjust to expected NN input
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            # transforms.CenterCrop(self.size), # we could use this instead for very large images!
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor() # maps 0 -> 0.0, 255 -> 1.0 which matches artificial dataset
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.seed is not None:
            unique_seed = self.seed + idx
            np.random.seed(unique_seed)
            torch.manual_seed(unique_seed)
            torch.cuda.manual_seed(unique_seed)

        path = self.images[idx]

        img = Image.open(path)

        target = self.transform(img)

        # Uses the same degradation model as artificial geometric dataset
        return apply_degradation(target=target, size=self.size,
                                 sigma=self.sigma, dark_map=self.dark_map,
                                 poisson_noise=self.poisson_noise, full_detector_model=self.full_detector_model,
                                 data_loading=self.data_loading)

