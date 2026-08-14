import argparse
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import librosa
from scipy.interpolate import interp1d
import sys
import pandas as pd
import h5py
from scipy.spatial.transform import Rotation as R
import torch
from transformers import Wav2Vec2Processor, HubertModel, WavLMModel
import torch.nn.functional as F




class AudioFeatureExtractor:
    def __init__(self, model_id):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = WavLMModel.from_pretrained(model_id).requires_grad_(False).eval().cuda()

    def __call__(self, raw_audio, seq_size=60):
        inputs = self.processor(raw_audio, sampling_rate=16000, return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            features = self.model(inputs).last_hidden_state
            features = F.interpolate(features.permute((0, 2, 1)), seq_size, mode='linear', align_corners=True).permute((0, 2, 1))
        return features.cpu().numpy()




SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]

def LocalToGlobalAA(local_aa, parent_idx):
    """
    local_aa: (N, 55 * 3)
    parent_idx: list of int, length 55
    """

    N = local_aa.shape[0]
    nJoints = len(parent_idx)
    local_rot = R.from_rotvec(local_aa.reshape(-1, 3)).as_matrix().reshape((N, nJoints, 3, 3))
    global_rot = np.zeros_like(local_rot)
    for i in range(nJoints):
        if parent_idx[i] == -1:
            global_rot[:, i] = local_rot[:, i]
        else:
            global_rot[:, i] = np.einsum('nij,njk->nik', global_rot[:, parent_idx[i]], local_rot[:, i])
    global_aa = R.from_matrix(global_rot.reshape(-1, 3, 3)).as_rotvec().reshape(N, -1)
    return global_aa



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
    

def create_h5(data_folder, h5_path, seq_size, stride_size, 
                local_rotation=True, 
                split='train',
                include_huberts=True,
                wavlm_model="patrickvonplaten/wavlm-libri-clean-100h-large"):
    smplx_data_folder = os.path.join(data_folder, "smplxflame_30")
    audio_data_folder = os.path.join(data_folder, "wave16k")
    sem_data_folder = os.path.join(data_folder, "sem")
    if split =='all':
        npz_files = glob(os.path.join(smplx_data_folder, "*.npz"))
    else:
        split_results = parse_train_test_split_csv(os.path.join(data_folder, "train_test_split.csv"))
        npz_files = [os.path.join(smplx_data_folder, f + ".npz") for f in split_results[split]]

    audio_feature_extractor = (
        AudioFeatureExtractor(wavlm_model) if include_huberts else None
    )

    h5_handler = h5py.File(h5_path, "w")

    dp_id = 0
    for npz_file in tqdm(npz_files):
        # pid
        pid = os.path.basename(npz_file).split('_')[0]
        pid = int(pid)

        # audio file
        audio_file = os.path.join(audio_data_folder, os.path.basename(npz_file).replace(".npz", ".wav"))
        sem_file = os.path.join(sem_data_folder, os.path.basename(npz_file).replace(".npz", ".txt"))
        raw_audio, sr = librosa.load(audio_file, sr=16000)
        raw_audio = raw_audio.astype(np.float32)

        # smplx coefficients
        smplx_data = np.load(npz_file, allow_pickle=True)
        betas = smplx_data["betas"].astype(np.float32)
        poses = smplx_data["poses"].astype(np.float32)
        expressions = smplx_data["expressions"].astype(np.float32)
        trans = smplx_data["trans"].astype(np.float32)

        num_frames = min(len(poses), int(len(raw_audio)/sr * 30))
        num_windows = (num_frames - seq_size) // stride_size + 1


        poses_lst = [poses[i * stride_size: i * stride_size + seq_size] for i in range(0, num_windows)]
        if local_rotation == False:
            poses_lst = [LocalToGlobalAA(poses_lst[i], parent_idx=SMPLX_PARENTS).astype(np.float32) for i in range(len(poses_lst))]
        expressions_lst = [expressions[i * stride_size: i *
                                       stride_size + seq_size] for i in range(0, num_windows)]
        trans_lst = [trans[i * stride_size: i * stride_size + seq_size]
                     for i in range(0, num_windows)]

        audio_stride = seq_size * sr // 30
        audio_lst = [
            raw_audio[(i * stride_size) * sr // 30: (i * stride_size ) * sr // 30 + audio_stride] for i in range(0, num_windows)
        ]

        for i in range(num_windows):
            assert len(poses_lst[i]) == seq_size
            assert len(expressions_lst[i]) == seq_size
            assert len(trans_lst[i]) == seq_size
            assert len(audio_lst[i]) == sr * seq_size // 30

            grp = h5_handler.create_group(f"{dp_id:010d}")
            grp.create_dataset("betas", data=betas, dtype=np.float32)
            grp.create_dataset("poses", data=poses_lst[i], dtype=np.float32)
            grp.create_dataset("expressions", data=expressions_lst[i], dtype=np.float32)
            grp.create_dataset("trans", data=trans_lst[i], dtype=np.float32)
            grp.create_dataset("pid", data=pid)
            grp.create_dataset("audio", data=audio_lst[i], dtype=np.float32)
            if include_huberts:
                huberts = audio_feature_extractor(audio_lst[i], seq_size=seq_size)
                grp.create_dataset("huberts", data=huberts, dtype=np.float32)
            dp_id = dp_id + 1

    h5_handler.close()


def create():
    parser = argparse.ArgumentParser(description="Create LMDB dataset")
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to the beat_english_v2.0.0 directory",
    )
    parser.add_argument("--seq_size", type=int, default="60",
                        help="Sequence size in frames")
    parser.add_argument("--stride_size", type=int,
                        default="20", help="Stride size in frames")
    parser.add_argument("--h5_path", type=str, default="",
                        help="Path to save the lmdb")
    parser.add_argument("--local_rotation", type=int, default="0",
                        help="Path to save the lmdb")
    parser.add_argument("--split", type=str, default="train", choices=['all', 'train', 'test', 'val', 'additional'], help="(train, val, test, additional)")
    parser.add_argument('--huberts', type=int, default='1')
    parser.add_argument(
        '--wavlm_model',
        type=str,
        default='patrickvonplaten/wavlm-libri-clean-100h-large',
        help='Hugging Face model id or local WavLM directory',
    )
    args = parser.parse_args()

    if args.h5_path == "":
        local_global = "local" if args.local_rotation!=0 else "global"
        args.h5_path = f"{args.split}_seq_size_{args.seq_size}_stride_size_{args.stride_size}_{local_global}.h5"

    if args.huberts==0:
        args.h5_path = args.h5_path.replace(".h5", "_no_huberts.h5")
    create_h5(
        data_folder=args.data_folder, 
        h5_path=args.h5_path, 
        seq_size=args.seq_size, 
        stride_size=args.stride_size,
        local_rotation=args.local_rotation!=0,
        split=args.split,
        include_huberts=args.huberts!=0,
        wavlm_model=args.wavlm_model,
    )






if __name__ == "__main__":
    create()
