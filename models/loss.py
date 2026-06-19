import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()

        window = (
            _2D_window
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(channel, 1, window_size, window_size)
            .contiguous()
        )

        return window

    def forward(self, img1, img2):

        channel = img1.size(1)
        window_size = 11

        window = self.create_window(window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=5, groups=channel)
        mu2 = F.conv2d(img2, window, padding=5, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=5, groups=channel)
            - mu1_sq
        )

        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=5, groups=channel)
            - mu2_sq
        )

        sigma12 = (
            F.conv2d(img1 * img2, window, padding=5, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = (
            (2 * mu1_mu2 + C1)
            * (2 * sigma12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1)
            * (sigma1_sq + sigma2_sq + C2)
        )

        return 1 - ssim.mean()


class CombinedLoss(nn.Module):

    def __init__(self,
                 l1_weight=0.8,
                 ssim_weight=0.2):

        super().__init__()

        self.l1 = nn.L1Loss()

        self.ssim = SSIMLoss()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, prediction, target):

        l1_loss = self.l1(prediction, target)

        ssim_loss = self.ssim(prediction, target)

        total_loss = (
            self.l1_weight * l1_loss
            +
            self.ssim_weight * ssim_loss
        )

        return total_loss
if __name__ == "__main__":

    criterion = CombinedLoss()

    pred = torch.rand(2, 3, 256, 256)

    target = torch.rand(2, 3, 256, 256)

    loss = criterion(pred, target)

    print(loss)