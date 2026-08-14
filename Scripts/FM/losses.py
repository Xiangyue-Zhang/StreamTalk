import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from einops import rearrange


class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """Compute the geodesic distance between two rotation matrices.

        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, min=-1 + 1e-6, max=1 - 1e-6)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction="mean"):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        else:
            raise RuntimeError(f"unsupported reduction: {reduction}")


def rotation_6d_to_matrix_torch(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class ReconsLoss(nn.Module):
    def __init__(self, rotation_channels, trans_channels, rotation_method="rot6d"):
        super(ReconsLoss, self).__init__()
        self.Loss = torch.nn.L1Loss(reduction='mean')
        self.rec_loss = GeodesicLoss()
        self.rotation_channels = rotation_channels
        self.trans_channels = trans_channels

        self.rotation_method = rotation_method

    def forward(self, motion_pred, motion_gt):
        assert len(motion_gt.shape) == 3
        assert len(motion_pred.shape) == 3
        if self.rotation_method == "rot6d":
            rotation_loss = 0
            motion_pred_r = rotation_6d_to_matrix_torch(motion_pred[:, :, :self.rotation_channels].reshape((-1, 6)))
            motion_gt_r = rotation_6d_to_matrix_torch(motion_gt[:, :, : self.rotation_channels].reshape((-1, 6)))
            rotation_loss = self.rec_loss(motion_pred_r, motion_gt_r)

            trans_loss = 0
            if self.trans_channels > 0:
                motion_pred_t = motion_pred[:, :, self.rotation_channels: self.rotation_channels + self.trans_channels]
                motion_gt_t = motion_gt[:, :, self.rotation_channels: self.rotation_channels + self.trans_channels]
                trans_loss = F.l1_loss(motion_pred_t, motion_gt_t)
            return rotation_loss + trans_loss
        else:
            return F.smooth_l1_loss(motion_pred, motion_gt)

    def forward_vel(self, motion_pred, motion_gt):  # 1
        loss = self.Loss(
            motion_pred[:, 1:] - motion_pred[:, :-1], motion_gt[:, 1:] - motion_gt[:, :-1])
        return loss

    def forward_smooth(self, motion_pred, motion_gt):

        B, L, C = motion_pred.shape
        pred = motion_pred[:, 0: L - 2] + \
            motion_pred[:, 2:L] - motion_pred[:, 1: L - 1] * 2
        gt = motion_gt[:, 0: L - 2] + \
            motion_gt[:, 2:L] - motion_gt[:, 1: L - 1] * 2
        loss = self.Loss(pred, gt)
        return loss



class ReconsLossWithMask(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rec_loss = GeodesicLoss()



    def forward(self, motion_pred, motion_gt, mask):
        assert len(motion_gt.shape) == 3
        assert len(motion_pred.shape) == 3

        B, L = motion_pred.shape[0], motion_pred.shape[1]
        motion_pred_r = rotation_6d_to_matrix_torch(motion_pred.reshape((-1, 6)))
        motion_gt_r = rotation_6d_to_matrix_torch(motion_gt.reshape((-1, 6)))
        rotation_loss = self.rec_loss(motion_pred_r, motion_gt_r, reduction="none").reshape((B, L, -1))
        nJonts = rotation_loss.shape[-1]
        mask = ~mask.unsqueeze(-1).expand((B, L, nJonts))
        rotation_loss = (rotation_loss * mask).sum() / (mask.sum() + 1e-5)
        return rotation_loss



class TraceLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, motion_pred, motion_gt, reduction="mean"):
        I = motion_gt @ motion_pred.transpose(-2, -1)
        res = 3 - (I[...,0,0] + I[...,1,1] + I[...,2,2])
        if reduction == "mean":
            return res.mean()
        elif reduction == "none" or reduction is None:
            return res
        else:
            raise RuntimeError(f"unsupported reduction: {reduction}")



class TraceLossWithMask(nn.Module):
    def __init__(self, ):
        super().__init__()



    def forward(self, motion_pred, motion_gt, mask):
        assert len(motion_gt.shape) == 3
        assert len(motion_pred.shape) == 3

        B, L = motion_pred.shape[0], motion_pred.shape[1]
        motion_pred_r = rotation_6d_to_matrix_torch(motion_pred.reshape((-1, 6)))
        motion_gt_r = rotation_6d_to_matrix_torch(motion_gt.reshape((-1, 6)))
        I = motion_gt_r @ motion_pred_r.transpose(2, 1)

        nJoints = 55
        rotation_loss = 3 - (I[:,0,0] + I[:,1,1] + I[:,2,2]).reshape((B, L, nJoints))
        mask = ~mask.unsqueeze(-1).expand((B, L, nJoints))
        rotation_loss = (rotation_loss * mask).sum() / (mask.sum() + 1e-5)
        return rotation_loss




class PosesReconsLoss(nn.Module):
    def __init__(self, rotation_channels, trans_channels):
        super(PosesReconsLoss, self).__init__()
        self.rotation_channels = rotation_channels
        self.trans_channels = trans_channels

    def forward(self, motion_pred, motion_gt):
        """
        motion_pred:B, L, C
        motion_gt:B, L, C
        """
        rotation_loss = 0
        motion_pred_r = rotation_6d_to_matrix_torch(motion_pred[:, :, :self.rotation_channels].reshape((-1, 6)))
        motion_gt_r = rotation_6d_to_matrix_torch(motion_gt[:, :, : self.rotation_channels].reshape((-1, 6)))
        I = motion_gt_r @ motion_pred_r.transpose(2, 1)
        rotation_loss = F.l1_loss(I, torch.eye(3, device=I.device).unsqueeze(0).expand((I.size(0), -1, -1)))

    
        trans_loss = 0
        if self.trans_channels > 0:
            motion_pred_t = motion_pred[:, :, self.rotation_channels: self.rotation_channels + self.trans_channels]
            motion_gt_t = motion_gt[:, :, self.rotation_channels: self.rotation_channels + self.trans_channels]
            trans_loss = F.l1_loss(motion_pred_t, motion_gt_t)
        return rotation_loss + trans_loss

