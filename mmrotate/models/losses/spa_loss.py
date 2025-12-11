from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from mmrotate.registry import MODELS


@MODELS.register_module()
class SPALoss(nn.Module):
    """Symmetric Prior Angle Loss (SPA Loss).

    For each predicted rotated bounding box, the box region is sampled via
    grid_sample and split into symmetric halves (horizontal/vertical).
    A similarity metric (SSIM/MS-SSIM/MSE) between mirrored halves is used
    as a symmetry prior to regularize angle prediction.

    Args:
        max_bboxes (int): Maximum number of bboxes per image used for SPA.
        loss_weight (float): Weight factor for the SPA loss.
        excluded_classes (list[int] | None): Class IDs ignored by SPA.
        grid_size (int): Resolution of per-box sampling grid.
        similarity_metric (str): One of {'ssim', 'msssim', 'mse'}.
    """

    def __init__(self,
                 max_bboxes: int = 100,
                 loss_weight: float = 1.0,
                 excluded_classes: Optional[List[int]] = None,
                 grid_size: int = 50,
                 similarity_metric: str = 'ssim') -> None:
        super().__init__()
        self.max_bboxes = max_bboxes
        self.loss_weight = loss_weight
        self.excluded_classes = excluded_classes
        self.grid = grid_size

        assert similarity_metric in {'ssim', 'msssim', 'mse'}, \
            f'Unsupported similarity_metric: {similarity_metric}'
        self.similarity_metric = similarity_metric

    # ------------------ Basic image similarity utilities ------------------ #

    def create_gaussian_kernel(self, window_size: int,
                               channel: int,
                               device: torch.device) -> Tensor:
        """Create a Gaussian kernel for SSIM computation."""

        def gaussian(window_size: int, sigma: float) -> Tensor:
            coords = np.arange(window_size) - window_size // 2
            gauss = np.exp(-(coords ** 2) / (2 * sigma ** 2))
            gauss = torch.from_numpy(gauss).float()
            return gauss / gauss.sum()

        sigma = 1.5
        _1d = gaussian(window_size, sigma).unsqueeze(1)  # (W, 1)
        _2d = _1d @ _1d.t()                               # (W, W)
        _2d = _2d.float().unsqueeze(0).unsqueeze(0)       # (1, 1, W, W)
        window = _2d.expand(channel, 1, window_size, window_size).to(device)
        # Normalization along spatial dimensions
        return window / window.sum(dim=(2, 3), keepdim=True)

    def ssim(self,
             img1: Tensor,
             img2: Tensor,
             window_size: int = 11,
             size_average: bool = True,
             full: bool = False) -> Tensor:
        """Compute SSIM between two images.

        Args:
            img1 (Tensor): (N, C, H, W).
            img2 (Tensor): (N, C, H, W).
        """
        img_x = img1.float()
        img_y = img2.float()

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        window = self.create_gaussian_kernel(window_size,
                                             channel=img_x.shape[1],
                                             device=img_x.device)

        mu_x = F.conv2d(img_x, window, padding=window_size // 2,
                        groups=img_x.shape[1])
        mu_y = F.conv2d(img_y, window, padding=window_size // 2,
                        groups=img_y.shape[1])

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(img_x * img_x, window,
                              padding=window_size // 2,
                              groups=img_x.shape[1]) - mu_x_sq
        sigma_y_sq = F.conv2d(img_y * img_y, window,
                              padding=window_size // 2,
                              groups=img_y.shape[1]) - mu_y_sq
        sigma_xy = F.conv2d(img_x * img_y, window,
                            padding=window_size // 2,
                            groups=img_x.shape[1]) - mu_xy

        numerator1 = 2 * mu_xy + c1
        numerator2 = 2 * sigma_xy + c2
        denominator1 = mu_x_sq + mu_y_sq + c1
        denominator2 = sigma_x_sq + sigma_y_sq + c2

        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)

        if size_average:
            ssim_val = ssim_map.mean()
        else:
            ssim_val = ssim_map.mean(dim=(1, 2, 3))

        if full:
            # The contrast_metric can be approximated simply using the sigma_xy term.
            contrast_metric = ((2.0 * sigma_xy + c2) /
                               (sigma_x_sq + sigma_y_sq + c2)).mean()
            return ssim_val, contrast_metric

        return ssim_val

    def msssim(self,
               img1: Tensor,
               img2: Tensor,
               levels: int = 5) -> Tensor:
        """Compute MS-SSIM between two images."""
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        msssim_vals = []
        x = img1
        y = img2

        for w in weights[:levels]:
            s = self.ssim(x, y)
            msssim_vals.append(w * s)

            # Stop if further downsampling is not possible.
            if min(x.shape[2], x.shape[3]) <= 1:
                break

            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

        return torch.stack(msssim_vals).sum()

    def mse(self, img1: Tensor, img2: Tensor) -> Tensor:
        return F.mse_loss(img1, img2)

    # ------------------ Core SPA loss on a single image ------------------ #

    def calculate_loss(self,
                       img: Tensor,
                       pred_bboxes: Tensor,
                       labels: Tensor,
                       max_bboxes: Optional[int] = None,
                       excluded_classes: Optional[List[int]] = None) -> Tensor:
        """Compute SPA loss for a single image.

        Args:
            img (Tensor): (C, H, W).
            pred_bboxes (Tensor): (N, 5) in (x, y, w, h, theta).
            labels (Tensor): (N,).
        """
        c, h, w = img.shape
        device = img.device
        n = pred_bboxes.shape[0]

        if n == 0:
            return torch.zeros((), dtype=img.dtype, device=device)

        # 1) Exclude specific classes (excluded_classes) from SPA computation
        if excluded_classes is not None and len(excluded_classes) > 0:
            excl = torch.as_tensor(excluded_classes,
                                   device=device,
                                   dtype=labels.dtype)
            mask = ~labels.unsqueeze(1).eq(excl).any(dim=1)
            pred_bboxes = pred_bboxes[mask]
            labels = labels[mask]
            n = pred_bboxes.shape[0]

            if n == 0:
                return torch.zeros((), dtype=img.dtype, device=device)

        # 2) If too many boxes exist, randomly sample a subset
        if max_bboxes is not None and n > max_bboxes:
            indices = torch.randperm(n, device=device)[:max_bboxes]
            pred_bboxes = pred_bboxes[indices]
            labels = labels[indices]
            n = max_bboxes

        # 3) Split coordinates and angle
        x = pred_bboxes[:, 0].detach()
        y = pred_bboxes[:, 1].detach()
        w_box = pred_bboxes[:, 2].detach()
        h_box = pred_bboxes[:, 3].detach()
        theta = pred_bboxes[:, 4]  # 각도는 gradient가 필요하므로 detach X

        # 4) Create box-local grid
        res = self.grid
        x_space = torch.linspace(-0.5, 0.5, steps=res, device=device)
        y_space = torch.linspace(-0.5, 0.5, steps=res, device=device)
        grid_y, grid_x = torch.meshgrid(y_space, x_space, indexing='ij')  # (R,R)

        grid_x = grid_x.unsqueeze(0).unsqueeze(-1)  # (1,R,R,1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1)  # (1,R,R,1)
        base_grid = torch.cat((grid_x, grid_y), dim=-1)  # (1,R,R,2)
        base_grid = base_grid.repeat(n, 1, 1, 1)         # (N,R,R,2)

        # 5) Apply width/height scaling
        w_box = w_box.view(n, 1, 1, 1)
        h_box = h_box.view(n, 1, 1, 1)
        scaled_grid = base_grid * torch.cat((w_box, h_box), dim=-1)  # (N,R,R,2)

        # 6) Apply rotation
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        rot_mat = torch.stack(
            [torch.stack([cos_t, -sin_t], dim=1),
             torch.stack([sin_t,  cos_t], dim=1)],
            dim=1,
        )  # (N,2,2)

        scaled_flat = scaled_grid.view(n, -1, 2).transpose(1, 2)  # (N,2,R*R)
        rotated_flat = torch.bmm(rot_mat, scaled_flat)            # (N,2,R*R)
        rotated_grid = rotated_flat.transpose(1, 2).view(n, res, res, 2)

        # 7) center translation (x,y)
        x = x.view(n, 1, 1, 1)
        y = y.view(n, 1, 1, 1)
        translated_grid = rotated_grid + torch.cat((x, y), dim=-1)  # (N,R,R,2)

        # 8) Normalize to [-1, 1]
        translated_grid[..., 0] = translated_grid[..., 0] / (w - 1) * 2 - 1
        translated_grid[..., 1] = translated_grid[..., 1] / (h - 1) * 2 - 1
        translated_grid = translated_grid.clamp(-1, 1)

        # 9) Extract patches using grid_sample (N,C,R,R)
        img_batch = img.unsqueeze(0).repeat(n, 1, 1, 1)
        patches = F.grid_sample(
            img_batch,
            translated_grid,
            mode='bicubic',
            padding_mode='zeros',
            align_corners=True,
        )

        # 10) Split into left/right or top/bottom halves to measure symmetry
        half = res // 2

        if self.similarity_metric == 'ssim':
            ssim_w = self.ssim(
                patches[:, :, :, :half],
                torch.flip(patches[:, :, :, half:], dims=[3]),
            )
            ssim_h = self.ssim(
                patches[:, :, :half, :],
                torch.flip(patches[:, :, half:, :], dims=[2]),
            )
            ssim_val = torch.max(ssim_w, ssim_h)
            return 1.0 - ssim_val

        if self.similarity_metric == 'msssim':
            msssim_w = self.msssim(
                patches[:, :, :, :half],
                torch.flip(patches[:, :, :, half:], dims=[3]),
            )
            msssim_h = self.msssim(
                patches[:, :, :half, :],
                torch.flip(patches[:, :, half:, :], dims=[2]),
            )
            msssim_val = torch.max(msssim_w, msssim_h)
            return 1.0 - msssim_val

        if self.similarity_metric == 'mse':
            mse_w = self.mse(
                patches[:, :, :, :half],
                torch.flip(patches[:, :, :, half:], dims=[3]),
            )
            mse_h = self.mse(
                patches[:, :, :half, :],
                torch.flip(patches[:, :, half:, :], dims=[2]),
            )
            return torch.min(mse_w, mse_h)

        raise RuntimeError('Unreachable similarity_metric branch')

    # ------------------ Public forward ------------------ #

    def forward(self,
                imgs: Tensor,
                pred_bboxes_list: List[Optional[Tensor]],
                label_list: List[Optional[Tensor]],
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Batch-wise SPA loss.

        Args:
            imgs (Tensor): (B, C, H, W).
            pred_bboxes_list (list[Tensor | None]): per-image bbox predictions.
            label_list (list[Tensor | None]): per-image labels.

        Returns:
            Tensor: scalar loss.
        """
        del avg_factor, reduction_override  # unused, for MMDet-style signature

        losses = []
        for img, pred_bboxes, labels in zip(imgs, pred_bboxes_list,
                                            label_list):
            if pred_bboxes is None or labels is None:
                continue

            loss_i = self.calculate_loss(
                img,
                pred_bboxes,
                labels,
                max_bboxes=self.max_bboxes,
                excluded_classes=self.excluded_classes,
            )
            losses.append(loss_i)

        if not losses:
            return torch.zeros((), dtype=imgs.dtype, device=imgs.device)

        return torch.stack(losses).mean() * self.loss_weight
