import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
# import kornia.geometry as G
from typing import Union
import math
import torch.nn as nn


def SplitIntoOverlapedWindows(seq, win_size, stride_step, keep_last=False):
    assert stride_step <= win_size
    overlap_size = win_size - stride_step
    num = (len(seq) - win_size) // stride_step + 1
    res = [seq[i * stride_step:i * stride_step + win_size] for i in range(num)]
    if keep_last:
        last_id = num * stride_step
        if last_id < len(seq):
            res.append(seq[last_id:])
    return res


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].reshape((matrix.shape[:-2] + (6,)))


def Normalize(x, eps=1e-12):
    len = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x / (len + eps)


# def NormalizeToRot6d(d):
#     origin_shape = d.shape
#     d6 = d.reshape((-1, 6))
#     a1, a2 = d6[..., :3], d6[..., 3:]
#     b1 = F.normalize(a1, dim=-1)
#     b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
#     b2 = F.normalize(b2, dim=-1)
#     d6 = torch.cat([b1, b2], dim=-1)
#     return d6.reshape(origin_shape)


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


def rotation_6d_to_matrix_numpy(d6: torch.Tensor):
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
    b1 = Normalize(a1)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = Normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def rotation_6d_to_matrix(d6: Union[torch.Tensor, np.ndarray]):
    if isinstance(d6, torch.Tensor):
        return rotation_6d_to_matrix_torch(d6=d6)
    elif isinstance(d6, np.ndarray):
        return rotation_6d_to_matrix_numpy(d6=d6)
    else:
        assert 0 == 1


def rotation_matrix_to_axis_angle_torch(rot_mat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    将旋转矩阵转换为轴角表示
    Args:
        rot_mat: (..., 3, 3) 旋转矩阵（必须有效正交矩阵）
        eps: 数值稳定性阈值
    Returns:
        axis_angle: (..., 3) 轴角表示（轴方向为单位向量，角度为向量的模长）
    """
    # 确保输入是3x3矩阵
    assert rot_mat.shape[-2:] == (3, 3), f"Invalid rotation matrix shape: {rot_mat.shape}"

    # 计算旋转角度θ（通过矩阵的迹）
    trace = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))
    # 计算轴方向
    axis = torch.empty_like(rot_mat[..., 0])
    # Case 1: θ ≈ 0（返回零向量）
    zero_angle = theta < eps
    axis[zero_angle] = torch.tensor([1.0, 0.0, 0.0], device=rot_mat.device)  # 默认轴

    # Case 2: θ ≈ π（需要特殊处理奇异性）
    near_pi = theta > math.acos(-1 + 2*eps)
    if torch.any(near_pi):
        # 从对角线元素计算轴
        r_diag = rot_mat[near_pi].diagonal(dim1=-2, dim2=-1)
        axis_near_pi = torch.sqrt( torch.clip((r_diag + 1) / 2, min=0.0) )
        # 符号需要根据非对角元素确定
        mask_x = ((rot_mat[near_pi, 2, 1] - rot_mat[near_pi, 1, 2]) > 0) * 2 - 1
        mask_y = ((rot_mat[near_pi, 0, 2] - rot_mat[near_pi, 2, 0]) > 0) * 2 - 1
        mask_z = ((rot_mat[near_pi, 1, 0] - rot_mat[near_pi, 0, 1]) > 0) * 2 - 1
        mask = torch.stack([mask_x, mask_y, mask_z], dim=-1)
        axis[near_pi] = axis_near_pi * mask


    # Case 3: 常规角度
    normal_angle = ~(zero_angle | near_pi)
    if torch.any(normal_angle):
        rot_mat_normal = rot_mat[normal_angle]
        axis[normal_angle] = torch.stack([
            rot_mat_normal[..., 2, 1] - rot_mat_normal[..., 1, 2],
            rot_mat_normal[..., 0, 2] - rot_mat_normal[..., 2, 0],
            rot_mat_normal[..., 1, 0] - rot_mat_normal[..., 0, 1]
        ], dim=-1) / (2 * torch.sin(theta[normal_angle].unsqueeze(-1)))

    # 将角度编码为向量的模长
    return axis * theta.unsqueeze(-1)


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    将向量转换为反对称矩阵。
    Args:
        v: (..., 3), 李代数向量
    Returns:
        K: (..., 3, 3), 反对称矩阵
    """
    K = torch.zeros((*v.shape[:-1], 3, 3), device=v.device, dtype=v.dtype)
    K[..., 0, 1] = -v[..., 2]
    K[..., 0, 2] = v[..., 1]
    K[..., 1, 0] = v[..., 2]
    K[..., 1, 2] = -v[..., 0]
    K[..., 2, 0] = -v[..., 1]
    K[..., 2, 1] = v[..., 0]
    return K

def axis_angle_to_rotation_matrix_torch(omega_vec, eps: float = 1e-6):
    """
    将李代数向量转换为旋转矩阵。
    Args:
        omega_vec: (..., 3), 李代数向量
    Returns:
        R: (..., 3, 3), 旋转矩阵
    """
    theta = torch.norm(omega_vec, dim=-1, keepdim=True)
    near_zero = theta.squeeze(-1) < eps
    k = hat(omega_vec)
    K = hat( omega_vec /  (theta + 1e-12))   
    I = torch.eye(3, device=omega_vec.device, dtype=omega_vec.dtype).repeat(*omega_vec.shape[:-1], 1, 1)
    theta_expanded = theta.unsqueeze(-1)
    sin_theta = torch.sin(theta_expanded)
    cos_theta = torch.cos(theta_expanded)

    # Rodrigues公式
    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)

    # 小角度近似
    R_approx = I + k + 0.5 * (k @ k)
    R = torch.where(near_zero[..., None, None], R_approx, R)
    return R


def axis_angle_to_rotation_matrix_np(axis_angle):
    aa = Rotation.from_rotvec(axis_angle, degrees=False)
    aa = aa.as_matrix()
    return aa

def rotation_matrix_to_axis_angle(m: Union[np.ndarray, torch.Tensor], eps: float = 1e-4):
    ori_shape = m.shape
    m  = m.reshape((-1, 3, 3))
    if isinstance(m, torch.Tensor):
        res =  rotation_matrix_to_axis_angle_torch(m, eps=eps)
    elif isinstance(m, np.ndarray):
        res =  Rotation.from_matrix(m).as_rotvec(degrees=False)
    else:
        assert 0 == 1
    res = res.reshape(ori_shape[:-1])
    return res

def axis_angle_to_rotation_matrix(axis_angle: Union[np.ndarray, torch.Tensor], eps: float = 1e-4):
    assert axis_angle.shape[-1] == 3
    ori_shape = axis_angle.shape
    axis_angle = axis_angle.reshape((-1, 3))
    if isinstance(axis_angle, torch.Tensor):
        res =  axis_angle_to_rotation_matrix_torch(axis_angle, eps=eps)
    elif isinstance(axis_angle, np.ndarray):
        res =  axis_angle_to_rotation_matrix_np(axis_angle)
    else:
        assert 0 == 1

    res = res.reshape(ori_shape[:-1] + (3, 3))
    return res


def rotation_6d_to_axis_angle(m: Union[np.ndarray, torch.Tensor], eps: float = 1e-4):
    orig_shape = m.shape
    m = rotation_6d_to_matrix(m.reshape((-1, 6)))
    return rotation_matrix_to_axis_angle(m, eps=eps).reshape(orig_shape[:-1] + (orig_shape[-1] // 6 * 3,))


def  axis_angle_to_rotation_6d(m: Union[np.ndarray, torch.Tensor]):
    orig_shape = m.shape
    m = axis_angle_to_rotation_matrix(m.reshape((-1, 3)))
    return matrix_to_rotation_6d(m).reshape(orig_shape[:-1] + (orig_shape[-1] // 3 * 6,))


def NormalizeRot6d(m):
    orig_shape = m.shape
    d6 = m.reshape((-1, 6))
    a1, a2 = d6[..., :3], d6[..., 3:]

    if isinstance(m, np.ndarray):
        b1 = Normalize(a1)
        b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 = Normalize(b2)
        m = np.concatenate([b1, b2], axis=-1)
    elif isinstance(m, torch.Tensor):
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        m = torch.cat([b1, b2], dim=-1)

    m = m.reshape(orig_shape)
    return m




def rotation_matrix_to_quaternion_torch(rot_mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数（w, x, y, z），支持 batch 维度
    
    参数:
        rot_mat: 旋转矩阵，形状为 (..., 3, 3)
        eps: 数值稳定性的小值
    
    返回:
        quat: 四元数，形状为 (..., 4)，顺序为 (w, x, y, z)
    """
    assert rot_mat.shape[-2:] == (3, 3), f"输入的最后两个维度必须是 (3, 3)，但得到 {rot_mat.shape[-2:]}"
    
    # 获取输入形状
    orig_shape = rot_mat.shape
    rot_mat = rot_mat.view(-1, 3, 3)
    batch_size = rot_mat.shape[0]
    
    # 计算矩阵的迹
    trace = torch.einsum('bii->b', rot_mat)
    
    # 初始化四元数输出
    quat = torch.zeros(batch_size, 4, device=rot_mat.device, dtype=rot_mat.dtype)
    
    # 情况1: 迹 > 0
    mask1 = trace > eps
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        w = 0.25 * s
        x = (rot_mat[mask1, 2, 1] - rot_mat[mask1, 1, 2]) / s
        y = (rot_mat[mask1, 0, 2] - rot_mat[mask1, 2, 0]) / s
        z = (rot_mat[mask1, 1, 0] - rot_mat[mask1, 0, 1]) / s
        quat[mask1] = torch.stack([w, x, y, z], dim=1)
    
    # 情况2: 迹 <= 0，需要找到对角线上的最大元素
    mask2 = ~mask1
    if mask2.any():
        # 找到每个矩阵中对角线元素的最大值及其索引
        diag = torch.stack([
            rot_mat[mask2, 0, 0],
            rot_mat[mask2, 1, 1], 
            rot_mat[mask2, 2, 2]
        ], dim=1)
        
        max_val, max_idx = torch.max(diag, dim=1)
        
        # 根据最大元素的索引选择不同的计算方式
        for i in range(3):  # 0, 1, 2 分别对应 xx, yy, zz 最大
            mask_i = (max_idx == i) & mask2[mask2]
            if not mask_i.any():
                continue
                
            # 提取对应的旋转矩阵
            R = rot_mat[mask2][mask_i]
            
            if i == 0:  # xx 最大
                s = torch.sqrt(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2] + eps) * 2
                w = (R[:, 2, 1] - R[:, 1, 2]) / s
                x = 0.25 * s
                y = (R[:, 0, 1] + R[:, 1, 0]) / s
                z = (R[:, 0, 2] + R[:, 2, 0]) / s
                
            elif i == 1:  # yy 最大
                s = torch.sqrt(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2] + eps) * 2
                w = (R[:, 0, 2] - R[:, 2, 0]) / s
                x = (R[:, 0, 1] + R[:, 1, 0]) / s
                y = 0.25 * s
                z = (R[:, 1, 2] + R[:, 2, 1]) / s
                
            else:  # i == 2, zz 最大
                s = torch.sqrt(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1] + eps) * 2
                w = (R[:, 1, 0] - R[:, 0, 1]) / s
                x = (R[:, 0, 2] + R[:, 2, 0]) / s
                y = (R[:, 1, 2] + R[:, 2, 1]) / s
                z = 0.25 * s
            
            quat[mask2][mask_i] = torch.stack([w, x, y, z], dim=1)
    
    # 确保四元数是单位四元数
    quat = nn.functional.normalize(quat, p=2, dim=1, eps=eps)
    
    # 恢复原始形状
    quat = quat.view(*orig_shape[:-2], 4)
    
    return quat



def slerp_torch(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    向量化版本的batch SLERP，支持多种输入形状
    """    
    # 计算点积
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    
    # 处理符号（选择最短路径）
    q2_adj = torch.where(dot < 0, -q2, q2)
    dot_adj = torch.abs(dot)
    
    # 限制点积范围
    dot_adj = torch.clamp(dot_adj, -1.0, 1.0)
    
    # 计算角度
    theta = torch.acos(dot_adj)
    sin_theta = torch.sin(theta)
    

    
    # 扩展t的形状以匹配q的维度
    t_expanded = t
    while t_expanded.dim() < q1.dim():
        t_expanded = t_expanded.unsqueeze(-1)


    # 数值稳定性处理
    epsilon = 1e-6
    safe_sin_theta = torch.maximum(sin_theta, torch.tensor(epsilon, device=sin_theta.device))
    # 计算插值系数
    scale1 = torch.sin((1.0 - t_expanded) * theta) / safe_sin_theta
    scale2 = torch.sin(t_expanded * theta) / safe_sin_theta
    
    flag =  sin_theta < epsilon
    scale1 = torch.where(flag, 1.0 - t_expanded, scale1)
    scale2 = torch.where(flag, t_expanded, scale2)

    # 插值并归一化
    result = scale1 * q1 + scale2 * q2_adj
    return F.normalize(result, p=2, dim=-1)



def quaternion_to_matrix_torch(quaternions):
    """
    将四元数转换为旋转矩阵（支持batch输入）
    
    参数:
        quaternions: (..., 4) 四元数，格式为[w, x, y, z]，其中w是实部
        
    返回:
        rotation_matrix: (..., 3, 3) 对应的旋转矩阵
    """
    if quaternions.size(-1) != 4:
        raise ValueError(f"输入四元数的最后一个维度必须是4，但得到的是{quaternions.shape[-1]}")
    
    # 归一化四元数
    quaternions = nn.functional.normalize(quaternions, p=2, dim=-1)
    
    # 提取四元数分量
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    
    # 计算平方项
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    
    # 计算交叉项
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # 构建旋转矩阵
    # 第一列
    m00 = w2 + x2 - y2 - z2
    m10 = 2 * (xy + wz)
    m20 = 2 * (xz - wy)
    
    # 第二列
    m01 = 2 * (xy - wz)
    m11 = w2 - x2 + y2 - z2
    m21 = 2 * (yz + wx)
    
    # 第三列
    m02 = 2 * (xz + wy)
    m12 = 2 * (yz - wx)
    m22 = w2 - x2 - y2 + z2
    
    # 组合成旋转矩阵
    rotation_matrix = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1)
    ], dim=-2)
    
    return rotation_matrix


def slerp_rotmat_torch(rot_1, rot_2, t):
    q1 = rotation_matrix_to_quaternion_torch(rot_1)
    q2 = rotation_matrix_to_quaternion_torch(rot_2)
    qt = slerp_torch(q1, q2, t)
    Rt = quaternion_to_matrix_torch(qt)
    return Rt


def slerp_rot6d_torch(rot6d_1, rot6d_2, t):
    origin_shape = rot6d_1.shape
    assert rot6d_1.shape == rot6d_1.shape
    rot6d_1 = rot6d_1.reshape(*origin_shape[:-1], -1, 6)
    rot6d_2 = rot6d_2.reshape(*origin_shape[:-1], -1, 6)
    while t.dim() < rot6d_1.dim():
        t = t.unsqueeze(-1)

    rot_1 = rotation_6d_to_matrix(rot6d_1)
    rot_2 = rotation_6d_to_matrix(rot6d_2)


    Rt = slerp_rotmat_torch(rot_1, rot_2, t)
    Rt = matrix_to_rotation_6d(Rt)
    Rt = Rt.reshape(origin_shape)
    return Rt

if __name__ == '__main__':
    import time
    x = Rotation.from_euler('xyz', [0, 0, 180-0.1], degrees=True).as_matrix()[None].astype(np.float32)
    x_tensor = torch.from_numpy(x)

    aa_tensor = rotation_matrix_to_axis_angle(x_tensor)
    print(aa_tensor)

    aa = rotation_matrix_to_axis_angle(x)
    print(aa)


    r_tensor = axis_angle_to_rotation_matrix(aa_tensor)
    r = axis_angle_to_rotation_matrix(aa)

    d = r_tensor@torch.from_numpy(r).float()

    d = rotation_matrix_to_axis_angle(d)
    print(torch.linalg.norm(d, dim=-1) * 180 / math.pi)