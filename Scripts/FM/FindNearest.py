import sys
from Common.SMPLX_Util import GetSMPLXModel, SplitPosesIntoSmplxParams
import numpy as np
import torch
import librosa
from scipy.spatial.transform import Rotation as R
import pandas as pd
from tqdm import tqdm
import os
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def LoadDataFromFiles(npz_file, smplx_model, device='cuda'):
    smplx_data = np.load(npz_file, allow_pickle=True)
    betas = smplx_data["betas"].astype(np.float32)
    poses = smplx_data["poses"].astype(np.float32)
    expressions = smplx_data["expressions"].astype(np.float32)
    trans = smplx_data["trans"].astype(np.float32)

    n_frames = poses.shape[0]
    betas = torch.from_numpy(np.repeat(betas[None, :], n_frames, axis=0)).to(device)
    expressions = torch.from_numpy(expressions).to(device)
    poses = torch.from_numpy(poses).to(device)
    trans = torch.from_numpy(trans).to(device)


    params = SplitPosesIntoSmplxParams(poses)
    params['global_orient'] = torch.zeros_like(params['global_orient'])
    with torch.no_grad():
        output = smplx_model(betas=betas,
                                **params,
                                transl=torch.zeros_like(trans),
                                expression=torch.zeros_like(expressions))
        joints = output['joints'].cpu().numpy()[:,:55]



    poses[:,:3] = 0

    if len(poses) != len(joints):
        print(npz_file)
        exit(-1)

    res = {
        "joints":joints,
        "poses":poses.cpu().numpy(),
        "trans": trans.cpu().numpy(),
        "betas": betas.cpu().numpy(),
    }
    return res

def MFCC(audio, sr=16000, n_mfcc=64):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,  hop_length=1050)
    mfcc = mfcc.T
    return mfcc



def parse_train_test_split_csv(csv_file):
    tables = pd.read_csv(csv_file)
    file_ids = tables['id']
    file_types = tables['type']
    train_files = []
    test_files =[]
    val_files = []
    additional_files = []
    for n in range(len(file_ids)):
        if file_types[n] == 'train':
            train_files.append(file_ids[n])
        elif file_types[n] == 'test':
            test_files.append(file_ids[n])
        elif file_types[n] == 'val':
            val_files.append(file_ids[n])
        elif file_types[n] == 'additional':
            additional_files.append(file_ids[n])
        else:
            assert 0 == 1
    return {
        "train":train_files,
        "test":test_files,
        "val":val_files,
        "traadditionalin":additional_files,
    }



def load_joints_from_files(pid='2_scott_0', 
                   data_folder=None,
                   csv_file=None,
                   smplx_model_path=None):
    if data_folder is None or csv_file is None or smplx_model_path is None:
        raise ValueError(
            'data_folder, csv_file, and smplx_model_path are required'
        )
    valid_ids = [npz_file for npz_file in parse_train_test_split_csv(csv_file)['train'] if npz_file.startswith(f'{pid}')]
    npz_files = [os.path.join(data_folder, npz_file + '.npz') for npz_file in valid_ids]



    device = 'cuda'
    smplx_model = (
        GetSMPLXModel(smplx_model_path)
        if smplx_model_path is not None
        else GetSMPLXModel()
    ).requires_grad_(False).to(device).eval()
    num_frames = 0


    res = []


    K = 0
    for npz_file in tqdm(npz_files, total=len(npz_files)):
        data = LoadDataFromFiles(npz_file, smplx_model, device=device)
        res.append(data)
        num_frames += data['joints'].shape[0]
        K += 1
        if K >= 200:
            break
    print(f'Total frames: {num_frames}')

    return res



def LoadTestData():
    npz_file = '/home/mm/Desktop/temp/ablation/ours/best_pid_2/res_2_scott_0_1_1.npz'

    device = 'cuda'
    smplx_model = GetSMPLXModel().requires_grad_(False).to(device).eval()
    data = LoadDataFromFiles(npz_file, smplx_model, device=device)
    return data


def MakeKDTree(
    pid='2_scott_0',
    data_folder=None,
    csv_file=None,
    smplx_model_path=None,
):
    data = load_joints_from_files(
        pid=pid,
        data_folder=data_folder,
        csv_file=csv_file,
        smplx_model_path=smplx_model_path,
    )
    if not data:
        raise RuntimeError(f'no retrieval motions found for speaker {pid}')

    all_joints = np.concatenate([d['joints'] for d in data], axis=0).astype(np.float32)

    all_poses = np.concatenate([d['poses'] for d in data], axis=0)
    all_trans = np.concatenate([d['trans'] for d in data], axis=0)

    
    feature_points = all_joints.reshape((all_joints.shape[0], -1))

    # feature_points = np.repeat(feature_points, 20, axis=0)



    kdtree = cKDTree(feature_points)


    return {
        'kdtree':kdtree,
        'poses': all_poses,
        'joints': all_joints.reshape(len(all_joints), -1),
    }



def FindNearest():
    test_data = LoadTestData()
    data = load_joints_from_files()

    all_joints = np.concatenate([d['joints'] for d in data], axis=0).astype(np.float32)
    all_poses = np.concatenate([d['poses'] for d in data], axis=0)
    all_trans = np.concatenate([d['trans'] for d in data], axis=0)

    

    kdtree = cKDTree(all_joints.reshape((all_joints.shape[0], -1)))

    
    nearest_poses = []
    nearest_distances = []
    for j in tqdm(test_data['joints'], total=test_data['joints'].shape[0]):
        dist, idx = kdtree.query(j.reshape(-1), k=1)
        nearest_poses.append(all_poses[idx])
        nearest_distances.append(dist)
    nearest_poses = np.stack(nearest_poses, axis=0)
    nearest_poses[:,:3] = 0
    

    print(f'Average distance: {np.mean(nearest_distances)}')
    plt.plot(nearest_distances)
    plt.show()
    np.savez(
        "nearest.npz",
        betas=test_data["betas"][0],
        poses=nearest_poses,
        expressions=np.zeros((nearest_poses.shape[0], 100), dtype=np.float32),
        trans=np.zeros((nearest_poses.shape[0], 3), dtype=np.float32),
        gender="neutral",
        mocap_frame_rate=np.array(30),
        model="smplx2020",
    )

    test_data['poses'][:,:3] = 0
    np.savez(
        "origin.npz",
        betas=test_data["betas"][0],
        poses=test_data['poses'],
        expressions=np.zeros((nearest_poses.shape[0], 100), dtype=np.float32),
        trans=np.zeros((nearest_poses.shape[0], 3), dtype=np.float32),
        gender="neutral",
        mocap_frame_rate=np.array(30),
        model="smplx2020",
    )

if __name__ == '__main__':
    FindNearest()
