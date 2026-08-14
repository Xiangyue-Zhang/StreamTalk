import sys
import numpy as np
import random
import torch.distributed as dist
import torch.utils.data.distributed
import torch
import os
import argparse
from einops import rearrange
from Common.BEATXDataset import H5Dataset
# from DiffusionNet import  MDMPartsTrans
# from Net import  MDMPartsTransCrossSeed as MotionModel
from Net import  DiffusionDITNetPartsFixedExpressions2PostNormInteraction2 as  MotionModel
from Net import SimpleSpeechModel
from torch.optim.lr_scheduler import StepLR
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from Common.Util import  rotation_6d_to_matrix, axis_angle_to_rotation_6d
from Common.SMPLXConfig import GetBodyPartIndices
from losses import ReconsLoss, ReconsLossWithMask, PosesReconsLoss
from Common.FK import GFKWraper
from Common.SMPLX_Util import SMPLX_PARENTS
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
def WriteObj(filename, verts, faces=None):
    with open(filename, 'w') as fh:
        for v in verts:
            fh.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if faces is not None:
            for f in faces:
                fh.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))



def build_prompt_mask(
    B: int,
    T: int,
    n_seed: int = 8,
    device = None,
    # 选择策略的概率：
    # 策略1：每隔k帧插1帧 ；策略3：最后n_tail帧全mask ；策略4：最后n_tail帧中随机1帧mask
    # 策略2：每隔k帧mask n帧，其概率 = 1 - p1 - p3 - p4
    p_strategy1: float = 0.25,
    p_strategy3: float = 0.25,
    p_strategy4: float = 0.25,
    # k 与 n 的采样范围（闭区间）
    k_range = (6, 12),
    n_range = (2, 6),
    ensure_n_le_k: bool = True,
    # 尾段长度（用于策略3/4）
    n_tail: int = 8,
    # 固定策略：1/2/3/4；None 表示按概率随机
    fixed_strategy = None,
) -> torch.Tensor:
    """
    返回:
        mask: (B, T) bool。False=使用模板/提示；True=由模型预测。

    统一规则:
      - 前 n_seed 帧恒为 False（seed）
      - 然后按样本选择以下四类策略之一：
        策略1：每隔 k 帧插入(遮) 1 帧          -> 条件 (t % k) == 0
        策略2：每隔 k 帧遮住连续 n 帧          -> 条件 (t % k) < n
        策略3：最后 n_tail 帧全部遮
        策略4：最后 n_tail 帧中随机遮 1 帧
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_seed = min(n_seed, T)
    # 基础：全 True，然后前 n_seed 置 False
    mask = torch.ones((B, T), dtype=torch.bool, device=device)
    if n_seed > 0:
        mask[:, :n_seed] = False
    if n_seed >= T:
        return mask  # 全在 seed 内，直接返回

    # —— 采样/设定策略编号：1/2/3/4（向量化） ——
    if fixed_strategy in (1, 2, 3, 4):
        s1 = torch.full((B, 1), fixed_strategy == 1, dtype=torch.bool, device=device)
        s3 = torch.full((B, 1), fixed_strategy == 3, dtype=torch.bool, device=device)
        s4 = torch.full((B, 1), fixed_strategy == 4, dtype=torch.bool, device=device)
        s2 = ~(s1 | s3 | s4)
    else:
        p1 = max(0.0, min(1.0, float(p_strategy1)))
        p3 = max(0.0, min(1.0, float(p_strategy3)))
        p4 = max(0.0, min(1.0, float(p_strategy4)))
        p2 = max(0.0, 1.0 - p1 - p3 - p4)  # 剩余给策略2
        # 采样多项式分布（用累积概率）
        u = torch.rand((B, 1), device=device)
        s1 = u < p1
        s3 = (~s1) & (u < (p1 + p3))
        s4 = (~s1) & (~s3) & (u < (p1 + p3 + p4))
        s2 = ~(s1 | s3 | s4)

    # —— 采样 k、n（仅策略1/2需要） ——
    k_low, k_high = k_range
    n_low, n_high = n_range
    k = torch.randint(k_low, k_high + 1, (B, 1), device=device)  # (B,1)
    n = torch.randint(n_low, n_high + 1, (B, 1), device=device)  # (B,1)
    if ensure_n_le_k:
        n = torch.minimum(n, k)

    # —— 有效时间轴（去掉 seed 段） ——
    L = T - n_seed
    t = torch.arange(L, device=device).view(1, -1)  # (1, L)，表示相对索引 n_seed..T-1

    # 策略1：每隔 k 帧遮 1 帧
    cond1 = (t % k) == 0  # (B, L)

    # 策略2：每隔 k 帧遮连续 n 帧
    cond2 = (t % k) < n   # (B, L)

    # 策略3：最后 n_tail 帧全部遮
    tail = max(0, min(int(n_tail), L))
    cond3 = (t >= (L - tail)) if tail > 0 else torch.zeros((1, L), dtype=torch.bool, device=device)

    # 策略4：最后 n_tail 帧随机遮 1 帧（每个样本各选1个尾部位置）
    if tail > 0:
        idx_tail = torch.randint(0, tail, (B,), device=device)                 # (B,)
        pos = (L - tail) + idx_tail                                            # (B,) 具体位置
        cond4 = torch.zeros((B, L), dtype=torch.bool, device=device)
        cond4[torch.arange(B, device=device), pos] = True                      # 置为 True=应遮
    else:
        cond4 = torch.zeros((B, L), dtype=torch.bool, device=device)

    # —— 合并四种策略（True=应遮=mask False） ——
    cond = (s1 & cond1) | (s2 & cond2) | (s3 & cond3) | (s4 & cond4)  # (B, L)

    # —— 写回整体 mask（前 n_seed 已经 False） ——
    mask[:, n_seed:] = ~cond  # cond=True → 应遮 → mask=False（用提示/模板）

    return mask

def TemporalStructureLoss(pred_bone_vec: torch.Tensor, target_bone_vec: torch.Tensor):
    assert pred_bone_vec.shape == target_bone_vec.shape
    assert len(pred_bone_vec.shape) == 4
    B, L, N, _ = pred_bone_vec.shape
    normalized_pred = pred_bone_vec / (torch.norm(pred_bone_vec, dim=-1, keepdim=True) + 1e-6)
    normalized_target = target_bone_vec / (torch.norm(target_bone_vec, dim=-1, keepdim=True) + 1e-6)

    pred_mat = torch.einsum('b l n k, b l m k-> b l n m', normalized_pred[:,:-1], normalized_pred[:,1:])
    target_mat = torch.einsum('b l n k, b l m k-> b l n m', normalized_target[:,:-1], normalized_target[:,1:])
    loss = torch.abs(pred_mat - target_mat)
    
    return loss.mean()

def SimilarityStructureLoss(pred: torch.Tensor, target: torch.Tensor):
    assert pred.shape == target.shape
    assert len(pred.shape) == 3
    B, L, D = pred.shape
    eps = 1e-6
    normalized_pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + eps)
    normalized_target = target / (torch.norm(target, dim=-1, keepdim=True) + eps)
    pred_mat = torch.einsum('b i k, b j k-> b i j', normalized_pred, normalized_pred)
    target_mat = torch.einsum('b i k, b j k-> b i j', normalized_target, normalized_target)
    loss = torch.abs(pred_mat - target_mat)
    
    return loss.mean()



def MakeMask(B, L, mask_ratio=0.2, device='cuda'):
    masked_id1 = torch.rand((B, L), device=device)
    sorted_values = torch.sort(masked_id1, dim=1).values
    masked_percentage = mask_ratio

    threshold_idx = int(masked_id1.shape[1] * masked_percentage)
    threshold = sorted_values[:, threshold_idx].reshape((masked_id1.shape[0], 1))
    masked_id1 = (masked_id1 < threshold)
    return masked_id1


def VEL(x: torch.Tensor):
    return x[:, 1:] - x[:, :-1]


def ACC(x: torch.Tensor):
    return VEL(VEL(x))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str,
                        default="../../Data/BEAT2/train_seq_size_60_stride_size_20_global.h5")

    parser.add_argument("--ckpt_folder", type=str, default="ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNormInteraction2-PoseRec-MultiMasks-debug")
    parser.add_argument("--batch_size", type=int, default="128")
    parser.add_argument("--num_workers", type=int, default="4")
    parser.add_argument("--epoch", type=int, default="1000")
    parser.add_argument("--lr", type=float, default="1e-4", help="i use 4e-5 previously")
    parser.add_argument("--local_rank", type=int,
                        help="local rank, will passed by ddp")
    parser.add_argument("--local-rank", type=int,
                        help="local rank, will passed by ddp")
    parser.add_argument("--sched_step", type=int, default="1000")
    parser.add_argument("--save_n_epoch", type=int, default="50")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--rotation_method", type=str, default="rot6d")
    args = parser.parse_args()
    setup_seed(args.seed)



    
    
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] is not None:
        print(os.environ["LOCAL_RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", rank=args.local_rank, init_method="env://")






    train_dataset = H5Dataset(h5_path=args.h5_path, rotation_method="axisangle", use_symmetry_aug=False) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, 
        prefetch_factor=10, drop_last=True, pin_memory=True, persistent_workers=True
    )

    # net = DiffusionDITNet(input_channels=55 * 6, hidden_size=1024).cuda()
    n_seed = 8
    net = MotionModel(hidden_size=384, num_person=31).cuda() 
    GFK = GFKWraper().cuda()
    simple_speech_model = SimpleSpeechModel().requires_grad_(False).eval().cuda()
    simple_speech_model.load_state_dict(torch.load("ckpt/split/SimpleSpeechModel/best.pt"))
    if args.resume != "":
        ckpt = torch.load(args.resume)
        net.load_state_dict(ckpt)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


    if dist.get_rank() == 0:
        log_dir = os.path.join(args.ckpt_folder, "log")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    hand_indices = GetBodyPartIndices("hand", "rot6d")
    upper_indices = GetBodyPartIndices("upper", "rot6d")
    lower_indices = GetBodyPartIndices("lower", "rot6d")
    face_joint_indices = GetBodyPartIndices("face", "rot6d")
    trans_indices = np.array([330, 331, 332])  # the last 3 dims are translation

        
    pose_loss_func2 = ReconsLossWithMask()
    pose_mse_func = PosesReconsLoss(55*6, 3)

    for epoch in range(args.epoch):
        net.train()
        if dist.get_rank() == 0:
            epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        batch_loss = 0
        batch_num = 0
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            poses = data['poses'].cuda()
            trans = data['trans'].cuda()
            poses = axis_angle_to_rotation_6d(poses)
            pid = data['pid'].cuda()
            huberts = data['huberts'].cuda()
            rest_joints = data['J'].cuda().unsqueeze(1).expand((-1, poses.shape[1], -1, -1)).reshape((-1, 55,  3))

            B,L  = poses.shape[:2]
            assert B >=32 and B % 8 ==0
            mask1 = torch.full((B - B//8, L), True, dtype=torch.bool, device=poses.device)
            # 前8帧不动
            mask1[:B//8,:n_seed] = False
            #前4帧不动
            mask1[B//8:B//8*2,:n_seed//2] = False
            #前4帧，后4帧不动
            mask1[B//8*2:B//8*3,:n_seed//2] = False
            mask1[B//8*2:B//8*3,-n_seed//2:] = False
            #前8帧，后8帧中的一帧
            mask1[B//8*3:B//8*4,:n_seed] = False
            mask1[B//8*3:B//8*4, L - n_seed + random.randint(0,7)] = False
            #前4帧，后4帧中的一帧
            mask1[B//8*4:B//8*5,:n_seed//2] = False
            mask1[B//8*4:B//8*5, L - n_seed//2 + random.randint(0,3)] = False
            # 前2帧，后2帧
            mask1[B//8*5:B//8*6,:n_seed//4] = False     
            mask1[B//8*5:B//8*6,-n_seed//4:] = False 
            # 前8帧，后随机一帧
            mask1[B//8*6:B//8*7,:n_seed] = False
            mask1[B//8*6:B//8*7, n_seed + random.randint(0,L-2*n_seed-1)] = False


            mask2 = MakeMask(B//8, L, mask_ratio=random.random()*0.3 + 0.5, device=poses.device)
            prompt_mask = torch.cat([mask1, mask2], dim=0)

            # prompt_mask = build_prompt_mask(B, L)

            trans_prompt = trans
            trans_mask = torch.full((B, L), True, dtype=torch.bool, device=poses.device)
            trans_mask[:, :n_seed] = False


            x1 = torch.cat([poses, trans], dim=-1)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=x1.device)
            sigma = 0.0
            xt = x0 * (1 - t[:,None, None] + sigma * t[:,None, None]) + x1 * t[:,None, None]
            pid_for_speech = pid.clone()
            pid_flag = pid > 30
            if torch.any(pid_flag):
                pid_for_speech[pid_flag] = pid_for_speech[pid_flag] - 30
            expressions = simple_speech_model(huberts, pid=pid_for_speech)
            model_kwargs = {"pid": pid,
                            "poses_prompt": poses,
                            "prompt_mask": prompt_mask,
                            "trans_prompt":trans_prompt,
                            "trans_mask":trans_mask,
                            "hand_indices": hand_indices,
                            "upper_indices": upper_indices,
                            "lower_indices": lower_indices,
                            "trans_indices": trans_indices,
                            "face_joint_indices": face_joint_indices,
                            "hubert": huberts,
                            "expressions": expressions
                            }

            pred_x1 = net(xt, t, **model_kwargs)

            # mse loss
            # originally i only used mse_loss = pose_mse_func(pred_x1, x1) 
            mse_loss = pose_mse_func(pred_x1, x1)
            # geodesic loss


            # geometric loss
            pred_trans = pred_x1[..., 330:333]
            gt_posemat = rotation_6d_to_matrix(poses.reshape((-1, 55, 6)))
            pred_posemat = rotation_6d_to_matrix(pred_x1[...,:330].reshape((-1, 55, 6)))
            vertices = GFK(gt_posemat, rest_joints).reshape((B, L, -1, 3))  + trans.unsqueeze(2)
            pred_vertices = GFK(pred_posemat, rest_joints).reshape((B, L, -1, 3)) + pred_trans.unsqueeze(2)
            # vertices = vertices[:,:, 55:]
            # pred_vertices = pred_vertices[:,:, 55:]
            vertices_loss = F.smooth_l1_loss(pred_vertices, vertices, beta=0.02)
            vel_loss = F.smooth_l1_loss(VEL(pred_vertices), VEL(vertices), beta=0.02)
            acc_loss = F.smooth_l1_loss(ACC(pred_vertices), ACC(vertices), beta=0.02)


            # geometric_loss = (vertices_loss + vel_loss + acc_loss) * 5.0  + (seed_vel_loss+ seed_acc_loss) * 500 # default weight is 1.0
            geometric_loss = (vertices_loss + (vel_loss + acc_loss)*10) * 1.0  # default weight is 1.0

            joint_bone_vec = (vertices[:,:,:55] - vertices[:, :, SMPLX_PARENTS])[:,:,1:]
            pred_joint_bone_vec = (pred_vertices[:,:,:55] - pred_vertices[:, :, SMPLX_PARENTS])[:,:,1:]

            # bone_vec = joint_bone_vec.reshape((-1, 3))
            # pred_bone_vec = pred_joint_bone_vec.reshape((-1, 3))
            # single_bone_loss = F.cosine_embedding_loss(pred_bone_vec, bone_vec,torch.ones((bone_vec.shape[0],), device=bone_vec.device))

            structure_loss = SimilarityStructureLoss(joint_bone_vec.reshape((-1, 54, 3)), pred_joint_bone_vec.reshape((-1, 54, 3)))


            # perception loss
            # pass
            pose_prompt_loss = pose_loss_func2(pred_x1[..., :330], poses, prompt_mask)* 1e-1
            trans_prompt_loss =( torch.abs(pred_trans -trans).sum(dim=-1) *(~trans_mask)).sum() / (~trans_mask).sum() * 1e-1

            
            
            loss = mse_loss +  geometric_loss + pose_prompt_loss +  structure_loss  + trans_prompt_loss

            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
            optimizer.step()

            if dist.get_rank() == 0:
                batch_loss = batch_loss + loss.item()
                batch_num = batch_num + 1

                if batch_idx % 20 == 0:
                    print(
                        f'EP {epoch} Loss {loss:.6f}  MSE {mse_loss:.6f} GM {geometric_loss:.6f} VER {vertices_loss:.6f} ST {structure_loss:.6f}  PP {pose_prompt_loss:.6f} TP {trans_prompt_loss:.6f}')
        if dist.get_rank() == 0:
            print(f'Epoch time {(time.time() - epoch_start_time)}')
            writer.add_scalar('train/loss', batch_loss / batch_num, epoch)
            if (epoch + 1) % args.save_n_epoch == 0:
                if not os.path.exists(args.ckpt_folder):
                    os.makedirs(args.ckpt_folder, exist_ok=True)
                torch.save(net.module.state_dict(), os.path.join(
                    args.ckpt_folder, ("epoch_%d") % (epoch,)))


if __name__ == "__main__":
    main()
