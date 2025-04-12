import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import config

from generate_trajectory import generate_trajectory_true


class datasets(Dataset):
    def __init__(self, data, lon_size, lat_size, max_seqlen):
        self.data = data
        self.w = lon_size
        self.h = lat_size
        self.max_seqlen = max_seqlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # # 生成模拟轨迹
        # traj = generate_trajectory_true(item, 2)
        # traj = traj[:, 4]
        # mask = torch.zeros(2)
        # mask[0] = 1

        V = self.data[item]
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        # 格网对应位置的编码
        grid_index = np.zeros((self.max_seqlen, 3, 2))
        for i in range(seqlen):
            location = (int(m_v[i, 0] * self.h), int(m_v[i, 1] * self.w))
            tmp = np.zeros((3, 2))
            tmp[0] = np.array([location[0] + 1, location[1] + 1])
            tmp[1] = np.array([location[0] - 1, location[1] - 1])
            tmp[2] = np.array(location)
            grid_index[i] = tmp
        grid_index[grid_index[:, :, 0] < 0, 0] = 0
        grid_index[grid_index[:, :, 0] == self.h, 0] = self.h - 1
        grid_index[grid_index[:, :, 1] < 0, 1] = 0
        grid_index[grid_index[:, :, 1] == self.w, 1] = self.w - 1

        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(int(V["mmsi"]), dtype=torch.int)
        time_start = torch.tensor(int(V["traj"][0, 4]), dtype=torch.int)
        grid_index = torch.tensor(grid_index, dtype=torch.int)
        return seq, grid_index, mask, seqlen, mmsi, time_start


if __name__ == '__main__':
    cf = config.Config('ct_dma')
    with open("data/ct_dma/ct_dma_train.pkl", 'rb') as f:
        data = pickle.load(f)
    da = datasets(data, cf.lon_size, cf.lat_size, cf.max_seqlen)
    for i in tqdm(range(len(data)), total=len(data)):
        seq, grid_index, mask, seqlen, mmsi, time_start = da[i]
    print(1)
