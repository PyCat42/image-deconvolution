import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure

def mse(prediction, target):
    """
    Calculates mean squared error.
    :param target:
    :param prediction:
    :return:
    """
    return torch.mean(torch.pow(target - prediction, 2))

def mae(prediction, target):
    """
    Calculate mean absolute error.
    More robust than MSE. Less sensitive to outliers.
    :param target:
    :param prediction:
    :return:
    """
    return torch.mean(torch.abs(target - prediction))

def psnr(prediction, target, max_val=1.0, eps=1e-8):
    """
    Calculates peak signal-to-noise ratio.
    Typically:
    * < 20dB - bad
    * 25 – 35dB - decent
    * 35dB - good reconstruction
    :param target:
    :param prediction:
    :param max_val:
    :param eps:
    :return:
    """
    return 10 * torch.log10((max_val ** 2) /(mse(target, prediction) + eps))

def ssim(prediction, target, device="cpu", max_val=1.0):
    """
    Calculates structural similarity index.
    Sensitive to blur. Captures edges and structures.
    :param target:
    :param prediction:
    :param max_val:
    :param device: "cuda" or "cpu" (default)
    :return:
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=max_val).to(device)
    return ssim(target, prediction)

class EdgeLoss(nn.Module):
    """
    Implements edge loss using sobel filters.
    """
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        )

        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        )

    def forward(self, prediction, target):
        pred_gx = F.conv2d(prediction, self.sobel_x, padding=1)
        pred_gy = F.conv2d(prediction, self.sobel_y, padding=1)

        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)

        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)

        return loss_x + loss_y

class HybridLoss(nn.Module):
    """
    Creates combination of MAE, SSIM, MSE and edge loss.
    """
    def __init__(self, device="cpu",
                 mae_share=0.7, edge_share=0.2,
                 ssim_share=0.1,  max_val=1.0,
                 mse_share=0.0):
        super().__init__()
        self.device = device
        self.mae_share = mae_share

        self.edge_share = edge_share
        self.edge_loss = EdgeLoss()

        self.ssim_share = ssim_share
        self.max_val = max_val

        self.mse_share = mse_share

        assert 0 <= self.mae_share <= 1
        assert 0 <= self.edge_share <= 1
        assert 0 <= self.ssim_share <= 1
        assert 0 <= self.mse_share <= 1
        assert self.mae_share + self.edge_share + self.ssim_share + self.mse_share <= 1

    def forward(self, prediction, target):
        # emphasizes pixelwise intensity tracking
        mae_loss = mae(prediction, target)

        # same as MAE but harder punishment for incorrect pixels
        mse_loss = mse(prediction, target)

        # emphasizes edges
        edge_term = self.edge_loss(prediction, target)

        # emphasizes structure
        # ssim = max_val means perfect => we need to minimize max_val - ssim
        ssim_score = ssim(prediction, target, device=self.device, max_val=self.max_val)
        ssim_loss = self.max_val - ssim_score

        hybrid_loss = (mae_loss * self.mae_share
                      + edge_term * self.edge_share
                      + ssim_loss * self.ssim_share
                      + mse_loss * self.mse_share)

        return hybrid_loss