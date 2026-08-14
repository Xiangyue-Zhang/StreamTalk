import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from .Util import axis_angle_to_rotation_6d
from .SMPLX_Util import  symmetrize_smplx_parameters
from  scipy.io import loadmat
import random
import json

class H5Dataset(Dataset):
    def __init__(self, 
                 h5_path, 
                 rotation_method="rot6d",
                 use_symmetry_aug=False):
        self.h5_handler = h5py.File(h5_path, "r")
        self.num = len(self.h5_handler.keys())

        self.rotation_method = rotation_method
        assert rotation_method =="rot6d" or rotation_method =="axisangle"

        self.rest_joint_position = {}
        for k, v in loadmat(os.path.join(os.path.dirname(__file__),'rest_joint_position.mat')).items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()

        self.use_symmetry_aug = use_symmetry_aug 
        if use_symmetry_aug:
            assert rotation_method == "axisangle", "symmetry augmentation only support axisangle"
                
    def __len__(self):
        return self.num




    def __getitem__(self, idx):
        grp = self.h5_handler[f"{idx:010d}"]
        betas = grp["betas"][:]
        poses_aa = grp["poses"][:]
        expressions = grp["expressions"][:]
        trans = grp['trans'][:]
        pid = int(grp['pid'][()])
        audio = grp['audio'][:]
        huberts = grp['huberts'][:]


        if self.rotation_method == "rot6d":
            poses = axis_angle_to_rotation_6d(poses_aa).astype(np.float32)
        elif self.rotation_method == "axisangle":
            poses = poses_aa
        
        betas = torch.from_numpy(betas)
        poses = torch.from_numpy(poses)
        expressions = torch.from_numpy(expressions)
        trans = torch.from_numpy(trans)
        huberts = torch.from_numpy(huberts)
        
        assert len(audio) == 16000 * len(poses)//30
        audio = torch.from_numpy(audio)
        
            
        J = self.rest_joint_position[pid]
        pid_offset = 0
        if self.use_symmetry_aug and random.random() < 0.5:
            poses, trans = symmetrize_smplx_parameters(poses, trans)
            pid_offset = 30
            J[..., 0] = -J[..., 0]
            
        return {
            "betas": betas,
            "poses": poses,
            "expressions": expressions,
            "trans":trans,
            "audio":audio,
            "huberts":huberts[0],
            "pid":pid + pid_offset,
            "J": J
        }



class H5DatasetStylization(Dataset):
    def __init__(self, 
                 h5_path, 
                 style_h5path,
                 rotation_method="rot6d"):
        self.h5_handler = h5py.File(h5_path, "r")
        self.num = len(self.h5_handler.keys())

        self.style_h5_handler = h5py.File(style_h5path, "r")
        with open(os.path.join(os.path.dirname(__file__),'id2motion_no_huberts.json'), "r") as f:
            self.id2motion = json.load(f)

        self.rotation_method = rotation_method
        assert rotation_method =="rot6d" or rotation_method =="axisangle"

        self.rest_joint_position = {}
        for k, v in loadmat(os.path.join(os.path.dirname(__file__),'rest_joint_position.mat')).items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()

                
    def __len__(self):
        return self.num




    def __getitem__(self, idx):
        grp = self.h5_handler[f"{idx:010d}"]
        betas = grp["betas"][:]
        poses_aa = grp["poses"][:]
        expressions = grp["expressions"][:]
        trans = grp['trans'][:]
        pid = int(grp['pid'][()])
        audio = grp['audio'][:]
        huberts = grp['huberts'][:]


        if self.rotation_method == "rot6d":
            poses = axis_angle_to_rotation_6d(poses_aa).astype(np.float32)
        elif self.rotation_method == "axisangle":
            poses = poses_aa
        
        betas = torch.from_numpy(betas)
        poses = torch.from_numpy(poses)
        expressions = torch.from_numpy(expressions)
        trans = torch.from_numpy(trans)
        huberts = torch.from_numpy(huberts)
        
        assert len(audio) == 16000 * len(poses)//30
        audio = torch.from_numpy(audio)
        
            
        J = self.rest_joint_position[pid]

        style_idx = np.random.randint(0, len(self.id2motion[str(pid)])-1)
        style_idx = self.id2motion[str(pid)][style_idx]
        style_grp = self.style_h5_handler[f"{style_idx:010d}"]
        style_poses = style_grp["poses"][:]

        if self.rotation_method == "rot6d":
            style_poses = axis_angle_to_rotation_6d(style_poses).astype(np.float32)


        return {
            "betas": betas,
            "poses": poses,
            'style_poses':style_poses,
            "expressions": expressions,
            "trans":trans,
            "audio":audio,
            "huberts":huberts[0],
            "pid":pid,
            "J": J
        }

class H5Dataset_no_huberts(Dataset):
    def __init__(self, 
                 h5_path, 
                 rotation_method="rot6d",
                 use_symmetry_aug=False):
        self.h5_handler = h5py.File(h5_path, "r")
        self.num = len(self.h5_handler.keys())

        self.rotation_method = rotation_method
        assert rotation_method =="rot6d" or rotation_method =="axisangle"

        self.rest_joint_position = {}
        for k, v in loadmat(os.path.join(os.path.dirname(__file__),'rest_joint_position.mat')).items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()

        self.use_symmetry_aug = use_symmetry_aug 
        if use_symmetry_aug:
            assert rotation_method == "axisangle", "symmetry augmentation only support axisangle"
        
        with open(os.path.join(os.path.dirname(__file__),'id2motion_no_huberts.json'), "r") as f:
            self.id2motion = json.load(f)

    def __len__(self):
        return self.num


    def get_one_with_pid(self, pid):
        style_idx = np.random.randint(0, len(self.id2motion[str(pid)])-1)
        style_idx = self.id2motion[str(pid)][style_idx]
        return self.__getitem__(int(style_idx))

    def __getitem__(self, idx):
        grp = self.h5_handler[f"{idx:010d}"]
        betas = grp["betas"][:]
        poses_aa = grp["poses"][:]
        expressions = grp["expressions"][:]
        trans = grp['trans'][:]
        pid = int(grp['pid'][()])
        audio = grp['audio'][:]



        if self.rotation_method == "rot6d":
            poses = axis_angle_to_rotation_6d(poses_aa).astype(np.float32)
        elif self.rotation_method == "axisangle":
            poses = poses_aa
        
        betas = torch.from_numpy(betas)
        poses = torch.from_numpy(poses)
        expressions = torch.from_numpy(expressions)
        trans = torch.from_numpy(trans)

        assert len(audio) == 16000 * len(poses)//30
        audio = torch.from_numpy(audio)
        
            
        J = self.rest_joint_position[pid]
        pid_offset = 0
        if self.use_symmetry_aug and random.random() < 0.5:
            poses, trans = symmetrize_smplx_parameters(poses, trans)
            pid_offset = 30
            J[..., 0] = -J[..., 0]
            
        return {
            "betas": betas,
            "poses": poses,
            "expressions": expressions,
            "trans":trans,
            "audio":audio,
            "pid":pid + pid_offset,
            "J": J
        }


def create_pid_list_json():
    from scipy.spatial.transform import Rotation as R
    from .SMPLX_Util import GlobalToLocalAA
    import soundfile as sf
    import json
    from tqdm import tqdm

    h5_path = "/mnt/4TDisk/mm_data/mm/project/PhysicallyPlausibleKeyPosesGuidedGeneration/Data/BEAT2/train_seq_size_300_stride_size_100_global_no_huberts.h5"
    ds = H5Dataset_no_huberts(h5_path, rotation_method="axisangle")

    N = len(ds)
    id2motion = {

    }
    for i in tqdm(range(N), total=N):
        data = ds[i]
        pid = data['pid']
        if pid not in id2motion:
            id2motion[pid] = []
        id2motion[pid].append(i)
    
    with open("id2motion_no_huberts.json", "w") as f:
        json.dump(id2motion, f)




# Example usage
if __name__ == "__main__":
    create_pid_list_json()

