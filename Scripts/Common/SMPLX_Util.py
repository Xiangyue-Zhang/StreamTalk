import smplx
import numpy as np
import torch

from .Util import *

def GetSMPLXModel(model_folder="/mnt/4TDisk/mm_data/mm/project/LargeMotionModel/Data/SMPLX_NEUTRAL_2020.npz"):
    model = smplx.create(model_folder, 
                         model_type="smplx",
                         gender="neutral", 
                         use_face_contour=False,
                         num_betas=300,
                         num_expression_coeffs=100,
                         ext="npz",
                         use_pca=False,
                         flat_hand_mean=True)
    return model



def SplitPosesIntoSmplxParams(poses):
    params = {}
    params["global_orient"] = poses[...,0:3]
    params["body_pose"] = poses[...,3:63 + 3]
    params["jaw_pose"] = poses[...,63 + 3:63 + 6]
    params["leye_pose"] = poses[...,63 + 6:63 + 9]
    params["reye_pose"] = poses[...,63 + 9:63 + 12] 
    params["left_hand_pose"] = poses[...,75:75 + 45] 
    params["right_hand_pose"] = poses[...,75 + 45:75 + 90]
    return params



SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]
def local_to_global_batch(local_rot, parents=SMPLX_PARENTS):
    """
    將局部旋轉轉換為全局旋轉。
    
    參數:
        local_rot (torch.Tensor): 局部旋轉矩陣，形狀為 (batch_size, num_joints, 3, 3)
        parents (list[int]): 父節點索引列表，長度為 num_joints
    
    返回:
        torch.Tensor: 全局旋轉矩陣，形狀同輸入
    """
    batch_size, num_joints, _, _ = local_rot.shape
    xp = torch if isinstance(local_rot, torch.Tensor) else np
    global_rot = xp.zeros_like(local_rot)
    
    for i in range(num_joints):
        if parents[i] == -1:
            # 根節點，直接使用局部旋轉
            global_rot[:, i] = local_rot[:, i]
        else:
            # 獲取父節點的全局旋轉並與當前局部旋轉相乘
            parent_rot = global_rot[:, parents[i]]  # (B, 3, 3)
            current_local = local_rot[:, i]        # (B, 3, 3)
            global_rot[:, i] = parent_rot @ current_local
    
    return global_rot






def global_to_local_batch(global_rot, parents=SMPLX_PARENTS):
    """
    將全局旋轉轉換為局部旋轉。
    
    參數:
        global_rot (torch.Tensor): 全局旋轉矩陣，形狀為 (batch_size, num_joints, 3, 3)
        parents (list[int]): 父節點索引列表，長度為 num_joints
    
    返回:
        torch.Tensor: 局部旋轉矩陣，形狀同輸入
    """
    batch_size, num_joints, _, _ = global_rot.shape
    xp = torch if isinstance(global_rot, torch.Tensor) else np
    local_rot = xp.empty_like(global_rot)
    
    for i in range(num_joints):
        if parents[i] == -1:
            # 根節點，局部旋轉等於全局旋轉
            local_rot[:, i] = global_rot[:, i]
        else:
            # 計算父節點旋轉的逆矩陣（轉置，因旋轉矩陣正交）
            parent_rot = global_rot[:, parents[i]]          # (B, 3, 3)
            if xp == torch:
                inv_parent_rot = parent_rot.transpose(-1, -2)   # (B, 3, 3)
            else:
                inv_parent_rot = np.transpose(parent_rot, axes=(0, 2, 1))
            current_global = global_rot[:, i]               # (B, 3, 3)
            local_rot[:, i] = inv_parent_rot @ current_global
    
    return local_rot


def LocalToGlobalAA(poses:np.ndarray):
    L,C = poses.shape
    mat = axis_angle_to_rotation_matrix(poses.reshape((-1, 3))).reshape((L,-1, 3,3))
    mat = local_to_global_batch(mat)
    aa = rotation_matrix_to_axis_angle(mat)
    return aa.reshape((L, C))


def GlobalToLocalAA(poses:np.ndarray):
    L,C = poses.shape
    mat = axis_angle_to_rotation_matrix(poses.reshape((L, -1, 3)))
    mat = global_to_local_batch(mat)
    aa = rotation_matrix_to_axis_angle(mat)
    return aa.reshape((L, C))





def symmetrize_smplx_parameters(pose_params, trans):
    """
    对称化 SMPLX 模型的姿态和形状参数
    
    参数:
        pose_params: 姿态参数, shape (N, 165) 或 (165,)
        beta_params: 形状参数, shape (N, 10) 或 (10,), 可选
        left_right_pairs: 左右关节对的列表, 可选
        
    返回:
        对称化后的姿态参数和形状参数
    """

    # SMPLX 标准左右关节对 (0-based 索引)
    left_right_pairs =( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body
                     (23,24), #eyes
                     (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51), (37,52), (38,53), (39,54) # hands
                    )
    
    # 确保输入是 torch.Tensor
    if not isinstance(pose_params, torch.Tensor):
        pose_params = torch.tensor(pose_params, dtype=torch.float32)
    if not isinstance(trans, torch.Tensor):
        trans = torch.tensor(trans, dtype=torch.float32)

    
    # 创建对称的姿态参数
    sym_pose = pose_params.clone()
    
    # 对每个左右对进行对称化
    for left, right in left_right_pairs:
        # 确保索引在有效范围内
        if left*3+3 <= sym_pose.shape[1] and right*3+3 <= sym_pose.shape[1]:
            # 交换左右参数并翻转旋转方向
            left_rot = sym_pose[:, left*3:left*3+3].clone()
            right_rot = sym_pose[:, right*3:right*3+3].clone()
            
            # 对称化: 左右交换并翻转某些轴的旋转
            sym_pose[:, left*3:left*3+3] = right_rot * torch.tensor([1, -1, -1], dtype=torch.float32)
            sym_pose[:, right*3:right*3+3] = left_rot * torch.tensor([1, -1, -1], dtype=torch.float32)
    
    sym_pose[:, 0:3] = pose_params[:, 0:3] * torch.tensor([1, -1, -1], dtype=torch.float32)
    sym_trans = trans.clone()
    sym_trans[...,0] *=-1
    return sym_pose, sym_trans



if __name__ =="__main__":
    # print(__path__)

    def WriteObj(filename, verts, faces=None):
        with open(filename, 'w') as fh:
            for v in verts:
                fh.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            if faces is not None:
                for f in faces:
                    fh.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


    smplx_model = GetSMPLXModel().requires_grad_(False).eval().cuda()
    from BEATXDataset import LMDBDataset
    lmdb_path = "/mnt/4TDisk/mm_data/mm/data/BEAT/V2/seq_size_60_stride_size_20.lmdb"
    lmdb_hubert = None
    ds = LMDBDataset(lmdb_path, lmdb_hubert, rotation_method="axisangle",)
    num_person = 0
    
    
    

    
    rest_joints_state = {}
    from tqdm import tqdm
    for data_id in tqdm(range(0, len(ds)), total=len(ds)):
        poses = ds[data_id]["poses"].cuda()
        pid = int(ds[data_id]["pid"])
        betas = ds[data_id]["betas"].cuda()
        shape_betas = betas.reshape((1, -1)).repeat((len(poses), 1))
        expressions = ds[data_id]["expressions"].cuda()
        trans = ds[data_id]["trans"].cuda()
        J = ds[data_id]["J"].cuda()

        # smplx_params = SplitPosesIntoSmplxParams(poses)
        # output = smplx_model(betas=None,
        #                     global_orient=smplx_params["global_orient"],
        #                     body_pose=smplx_params["body_pose"],
        #                     expression=expressions,
        #                     jaw_pose=smplx_params["jaw_pose"],
        #                     leye_pose=smplx_params["leye_pose"],
        #                     reye_pose=smplx_params["reye_pose"],
        #                     left_hand_pose=smplx_params["left_hand_pose"],
        #                     right_hand_pose=smplx_params["right_hand_pose"],
        #                     transl=trans,
        #                     return_verts=True)

        output = smplx_model()
        print(output['joints'].shape)
        print(output['vertices'].shape)
        
        # WriteObj("test.obj", output['vertices'][0].cpu().numpy(), smplx_model.faces)
        print(output['joints'][0][:55].cpu().numpy()[0])
        break
    from scipy.io import loadmat

    WriteObj("J.obj", J.cpu().numpy())
       

    
