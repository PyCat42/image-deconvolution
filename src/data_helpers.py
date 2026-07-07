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

        # Apply blur
        # rather than keeping blur kernel fixed sample it uniformally
        # + different amount of blur along x- and y-axis
        if self.sigma is None:
            sigma_x = np.random.uniform(1.0, 10.0)
            sigma_y = np.random.uniform(1.0, 10.0)
            sigma_max = max(sigma_x, sigma_y)
        else:
            sigma_x = self.sigma
            sigma_y = self.sigma
            sigma_max = self.sigma

        blur_kernel_size = int(np.ceil(6 * sigma_max))  # so we don't get truncated gaussian
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1  # blur kernel needs to be odd
        blurred = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), sigmaX=sigma_x, sigmaY=sigma_y)
        input_blur = torch.from_numpy(blurred).unsqueeze(dim=0)
        input = torch.clamp(input_blur, 0.0, 1.0)

        # get PSF (needed as input for RL algorithm)
        g_x = cv2.getGaussianKernel(blur_kernel_size, sigma_x)
        g_y = cv2.getGaussianKernel(blur_kernel_size, sigma_y)
        psf = g_y @ g_x.T
        psf = psf / psf.sum()

        photon_flux = 0.0
        if self.poisson_noise or self.full_detector_model:
            # Inject additional noise
            if self.photon_flux is None:
                photon_flux = np.random.uniform(20.0, 100.0)
            else:
                photon_flux = self.photon_flux
            counts = torch.poisson(input * photon_flux)
            input = counts.float() / photon_flux

        if self.full_detector_model:
            # Model dark counts - random, per pixel, per frame
            dark_counts = torch.full_like(input, np.random.uniform(0.01, 0.1))

            # Background - external light source
            background = np.random.uniform(0, 0.05)

            # Hot pixels
            hot_pixels = torch.zeros_like(input)
            num_hot_pixels = np.random.randint(10, 50)
            x = np.random.randint(0, self.size, size=num_hot_pixels)
            y = np.random.randint(0, self.size, size=num_hot_pixels)

            vals = torch.from_numpy(np.random.uniform(0.2, 0.6, size=num_hot_pixels).astype(np.float32))

            hot_pixels[0, y, x] = vals

            # Create final input
            input = torch.clamp(
                input + dark_counts + hot_pixels + self.dark_map + background,
                min=0, max=1
            )

        if self.data_loading:
            return input, target
        else:
            return input, target, psf, sigma_x, sigma_y, blur_kernel_size, photon_flux