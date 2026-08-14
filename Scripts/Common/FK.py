from .SMPLX_Util import SMPLX_PARENTS
import numpy as np
import torch

def WriteObj(filename, verts, faces=None):
    with open(filename, 'w') as fh:
        for v in verts:
            fh.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if faces is not None:
            for f in faces:
                fh.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


def regular_dodecahedron_vertices():
    # 黄金比例
    phi = (1 + np.sqrt(5)) / 2
    
    # 正十二面体的顶点坐标（边长为2/φ，需要缩放）
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ])
    
    # 计算当前边长（任意相邻两点间的距离）
    current_edge_length = np.linalg.norm(vertices[0] - vertices[1])
    
    # 缩放顶点使边长为1
    scale_factor = 1 / current_edge_length
    vertices *= scale_factor
    
    return vertices


        
def GFK_optimized(R: torch.Tensor, pos: torch.Tensor, parents, bone_vec=None):
    """
    R: [B, N, 3, 3] - 全局旋转矩阵
    pos: [B, N, 3] - rest position relative to parent
    parents: [N] - 父节点索引
    """
    B, N, _, _ = R.shape
    device = R.device

    # 预分配全局位置张量
    gpos = torch.zeros(B, N, 3, device=device)
    # gpos[:, 0] = pos[:,0]  # 根节点
    gpos[:, 0] = 0.0  # impose the root position to be 0

    # 预先计算所有骨骼向量 [N, 3]
    if pos is not None:
        bone_vec = pos - pos[:,parents]  # pos[:, parents, :] 是父节点的位置
    bone_vec[:,0] = 0  # 根节点的骨骼向量为0（可选）


    # 批量计算全局位置
    for i in range(1, N):
        parent_R = R[:, parents[i]]  # [B, 3, 3]
        rotated_vec = torch.einsum("bij,bj->bi", parent_R, bone_vec[:,i])
        gpos[:, i] = rotated_vec + gpos[:, parents[i]]

    return gpos  

class GFKWraper(torch.nn.Module):
    def __init__(self, n_virtual_joints=6, parents=SMPLX_PARENTS):
        super().__init__()

        local_virtual_joints, virtual_parent_idx = GFKWraper.create_virtual_joints(parents, n_virtual_joints)
        self.register_buffer("parents", torch.LongTensor(parents))
        self.register_buffer("virtual_parent_idx", torch.LongTensor(virtual_parent_idx))
        self.register_buffer("local_virtual_joints", torch.from_numpy(local_virtual_joints))
        self.n_virtual_joints = n_virtual_joints
        
    def forward(self, R: torch.Tensor, J: torch.Tensor=None, bone_vec: torch.Tensor=None):
        """
        R: [B, N, 3, 3], global rotation matrix
        J: [B, N, 3], rest joint positions
        """
        gpos = GFK_optimized(R, J, self.parents, bone_vec=bone_vec)
        parent_R = R[:, self.virtual_parent_idx] # [B, V, 3, 3]
        bone_vec = self.local_virtual_joints[None].expand(R.shape[0], -1, -1) # [B, V, 3]

        rotated_vec = torch.einsum("bnij,bnj->bni", parent_R, bone_vec)
        gpos_others = rotated_vec + gpos[:, self.virtual_parent_idx]
        gpos = torch.cat([gpos, gpos_others], dim=1)
        return gpos
    
    @staticmethod
    def create_virtual_joints(parents, n_virtual_joints=6):
        bone_length = 0.1
        if n_virtual_joints == 6:
            box = np.array([
                [-0.5,0,0],
                [0.5,0,0],
                [0, -0.5, 0],
                [0, 0.5, 0],
                [0,0,-0.5],
                [0,0,0.5],
            ], dtype=np.float32) * bone_length
        elif n_virtual_joints == 4:
            a = 1 / (2 * np.sqrt(2))
            box = np.array([
                [a, a, a],
                [-a, -a, a],
                [-a, a, -a],
                [a, -a, -a]
            ], dtype=np.float32) * bone_length
        elif n_virtual_joints == 20:
            box = regular_dodecahedron_vertices().astype(np.float32) * bone_length
        else:
            assert 0 == 1

        virtual_joints = []
        parent_idx = []
        for i in range(len(parents)):
            parent_idx = parent_idx + [i] * len(box)
            virtual_joints.append(box)
        virtual_joints = np.concatenate(virtual_joints, axis=0)
        
        return virtual_joints, parent_idx
    @staticmethod
    def _get_leaf_nodes(parents):
        flag = [0] * len(parents)
        for i in range(1, len(parents)):
            flag[parents[i]] = 1
        leaf_nodes = []
        for i in range(len(flag)):
            if flag[i] == 0:
                leaf_nodes.append(i)
        return leaf_nodes


    def IK(self, vertices: torch.Tensor):
        Rs = []
        Ts = [vertices[:, 0:4].mean(dim=1)]
        bone_vec = self.joints - self.joints [self.parents]  # pos[:, parents, :] 是父节点的位置
        bone_vec[0] = 0
        bone_vec = bone_vec[55:]
        for i in range(len(SMPLX_PARENTS)):
            child_vertices = vertices[:, i*4:(i+1)*4]
            child_rest_vertices = bone_vec[i*4:(i+1)*4].unsqueeze(0).repeat((child_vertices.shape[0], 1, 1))
            from_v = child_rest_vertices
            to_v = child_vertices - Ts[i].unsqueeze(1)
            R, T = compute_rigid_transform(from_v, to_v)
            Rs.append(R)
            Ts.append(T)
        Rs = torch.stack(Rs, dim=1)
        Ts = Ts[1]
        return Rs, Ts
            


def LFK_optimized(R: torch.Tensor, pos: torch.Tensor, parents):
    """
    R: [B, N, 3, 3] - local rotation matrix
    pos: [B, N, 3] - rest position relative to parent
    parents: [N] - 父节点索引
    """
    B, N, _, _ = R.shape
    device = pos.device


    # 预先计算所有骨骼向量 [B, N, 3]
    bone_vec = pos - pos[:,parents]  # pos[:, parents, :] 是父节点的位置
    bone_vec[:,0] = 0  # 根节点的骨骼向量为0（可选）

    T = torch.zeros(B, N, 4, 4, device=device)
    T[:,:,:3, :3] = R
    T[:,:,3, 3] = 1
    T[:,:,:3, 3] = bone_vec


    # updated_T = torch.zeros(B, N, 4, 4, device=device)
    # updated_T[:, 0] = T[:, 0]  # 根节点的变换矩阵
    # # 批量计算全局位置
    # for i in range(1, N):
    #     assert parents[i] < i, "Parents should be in a topological order"
    #     parent_T = updated_T[:, parents[i]]  # [B, 3, 3]
    #     updated_T[:, i] = parent_T @ T[:, i] 



    updated_T = [T[:, 0]] 
    # 批量计算全局位置
    for i in range(1, N):
        assert parents[i] < i, "Parents should be in a topological order"
        parent_T = updated_T[parents[i]]  # [B, 3, 3]
        updated_T.append(parent_T @ T[:, i]) 

    return torch.stack(updated_T, dim=1)  # [B, N, 4, 4]

class LFKWraper(torch.nn.Module):
    def __init__(self):
        super().__init__()

        local_virtual_joints, virtual_parent_idx = GFKWraper.create_virtual_joints(SMPLX_PARENTS)
        self.register_buffer("parents", torch.LongTensor(SMPLX_PARENTS))
        self.register_buffer("virtual_parent_idx", torch.LongTensor(virtual_parent_idx))
        self.register_buffer("local_virtual_joints", torch.from_numpy(local_virtual_joints))
        
    def forward(self, R: torch.Tensor, J: torch.Tensor):
        """
        R: [B, N, 3, 3], global rotation matrix
        J: [B, N, 3], rest joint positions
        """
        T = LFK_optimized(R, J, self.parents)
        gpos = T[:, :, :3, 3]  # [B, N, 3]
        parent_T = T[:, self.virtual_parent_idx] # [B, V, 4, 4]
        bone_vec = self.local_virtual_joints[None].expand(R.shape[0], -1, -1) # [B, V, 3]

        rotated_vec = torch.einsum("bnij,bnj->bni", parent_T[...,:3,:3], bone_vec)
        gpos_others = rotated_vec + gpos[:, self.virtual_parent_idx]
        gpos = torch.cat([gpos, gpos_others], dim=1)
        return gpos
    
    @staticmethod
    def create_virtual_joints(parents):
        bone_length = 0.1

        # box = np.array([
        #     [-0.5,0,0],
        #     [0.5,0,0],
        #     [0, -0.5, 0],
        #     [0, 0.5, 0],
        #     [0,0,-0.5],
        #     [0,0,0.5],
        # ], dtype=np.float32) * bone_length
        a = 1 / (2 * np.sqrt(2))
        box = np.array([
            [a, a, a],
            [-a, -a, a],
            [-a, a, -a],
            [a, -a, -a]
        ], dtype=np.float32) * bone_length

        virtual_joints = []
        parent_idx = []
        for i in range(len(parents)):
            parent_idx = parent_idx + [i] * len(box)
            virtual_joints.append(box)
        virtual_joints = np.concatenate(virtual_joints, axis=0)
        
        return virtual_joints, parent_idx
    @staticmethod
    def _get_leaf_nodes(parents):
        flag = [0] * len(parents)
        for i in range(1, len(parents)):
            flag[parents[i]] = 1
        leaf_nodes = []
        for i in range(len(flag)):
            if flag[i] == 0:
                leaf_nodes.append(i)
        return leaf_nodes


    def IK(self, vertices: torch.Tensor):
        Rs = []
        Ts = [vertices[:, 0:4].mean(dim=1)]
        bone_vec = self.joints - self.joints [self.parents]  # pos[:, parents, :] 是父节点的位置
        bone_vec[0] = 0
        bone_vec = bone_vec[55:]
        for i in range(len(SMPLX_PARENTS)):
            child_vertices = vertices[:, i*4:(i+1)*4]
            child_rest_vertices = bone_vec[i*4:(i+1)*4].unsqueeze(0).repeat((child_vertices.shape[0], 1, 1))
            from_v = child_rest_vertices
            to_v = child_vertices - Ts[i].unsqueeze(1)
            R, T = compute_rigid_transform(from_v, to_v)
            Rs.append(R)
            Ts.append(T)
        Rs = torch.stack(Rs, dim=1)
        Ts = Ts[1]
        return Rs, Ts


def compute_rigid_transform(xi: torch.Tensor, yi: torch.Tensor):
    """
    计算从点集{xi}到{yi}的最优旋转矩阵R和平移向量t，支持批量输入
    Args:
        xi: 输入点云，形状为(batch_size, num_points, 3)
        yi: 目标点云，形状与xi相同
    Returns:
        R: 最优旋转矩阵，形状为(batch_size, 3, 3)
        t: 最优平移向量，形状为(batch_size, 3)
    """
    # 确保输入为浮点类型
    
    assert xi.dtype == torch.float32 and yi.dtype == torch.float32, "Inputs must be float32"
    
    # 计算质心
    x_centroid = torch.mean(xi, dim=1, keepdim=True)  # (B, 1, 3)
    y_centroid = torch.mean(yi, dim=1, keepdim=True)  # (B, 1, 3)
    
    # 中心化点云
    xi_centered = xi - x_centroid
    yi_centered = yi - y_centroid
    
    # 计算协方差矩阵H
    H = torch.bmm(xi_centered.transpose(1, 2), yi_centered)  # (B, 3, 3)
    
    # SVD分解
    U, S, Vh = torch.linalg.svd(H)  # U: (B, 3, 3), Vh: (B, 3, 3)
    V = Vh.transpose(-1, -2)  # 转换为右奇异向量矩阵
    
    # 计算行列式符号调整
    det = torch.det(torch.bmm(V, U.transpose(1, 2)))  # (B,)
    sign_det = torch.sign(det)  # (B,)
    
    # 构建调整矩阵
    eye = torch.eye(3, device=xi.device).unsqueeze(0).repeat((len(xi),1,1))
    eye[:, 2, 2] = sign_det
    V_corrected = torch.bmm(V, eye)
    
    # 计算旋转矩阵
    R = torch.bmm(V_corrected, U.transpose(1, 2))
    
    # 计算平移向量
    t = y_centroid.squeeze(1) - torch.bmm(R, x_centroid.transpose(1, 2)).squeeze(-1)
    
    return R, t
  


if __name__ == "__main__":
    from Util import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
    from SMPLX_Util import GetSMPLXModel,SplitPosesIntoSmplxParams, LocalToGlobalAA
    from scipy.spatial.transform import Rotation as R
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    
    from BEATXDataset import LMDBDataset
    lmdb_path = "/mnt/4TDisk/mm_data/mm/data/BEAT/V2/seq_size_60_stride_size_20.lmdb"
    lmdb_hubert = None
    ds = LMDBDataset(lmdb_path, lmdb_hubert, rotation_method="axisangle",)
    smplx_model = GetSMPLXModel().requires_grad_(False).eval().cuda()
    data_id = 0
    poses = ds[data_id]["poses"].cuda()
    pid = int(ds[data_id]["pid"])
    betas = ds[data_id]["betas"].cuda()
    shape_betas = betas.reshape((1, -1)).repeat((len(poses), 1))
    expressions = ds[data_id]["expressions"].cuda()
    smplx_params = SplitPosesIntoSmplxParams(poses)
    trans = ds[data_id]["trans"].cuda()
    rest_joints = ds[data_id]["J"].cuda()

    output = smplx_model(betas=betas[None].expand((poses.shape[0], -1)),
                        global_orient=smplx_params["global_orient"],
                        body_pose=smplx_params["body_pose"],
                        expression=expressions,
                        jaw_pose=smplx_params["jaw_pose"],
                        leye_pose=smplx_params["leye_pose"],
                        reye_pose=smplx_params["reye_pose"],
                        left_hand_pose=smplx_params["left_hand_pose"],
                        right_hand_pose=smplx_params["right_hand_pose"],
                        # transl=trans,
                        return_verts=True)
    
    global_aa = LocalToGlobalAA(poses).reshape((-1, 55, 3))
    global_rot = axis_angle_to_rotation_matrix(global_aa)

    J = output['joints'].cuda()[:,:55]
    WriteObj("J.obj", J[0].cpu().numpy()[:55])

    fk = GFKWraper().cuda()


    pos = fk.forward(global_rot, rest_joints[None].expand((global_rot.shape[0], -1, -1)))
    print(pos.shape, rest_joints[None].expand((global_rot.shape[0], -1, -1)).shape)
    WriteObj("GJ.obj", (pos[:,:55] + rest_joints[None].expand((global_rot.shape[0], -1, -1))[:,[0]])[0].cpu().numpy())
    
    
    
    lk = LFKWraper().cuda()
    gpos = lk(axis_angle_to_rotation_matrix(poses.reshape((60, 55, 3))),
                  rest_joints[None].expand((poses.shape[0], -1, -1)))
    print(gpos.shape)
    WriteObj("LGJ.obj", (gpos + rest_joints[None].expand((global_rot.shape[0], -1, -1))[:,[0]])[0,:].cpu().numpy())

