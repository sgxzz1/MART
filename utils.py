# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for GPTrajectory.

References:
    https://github.com/karpathy/minGPT
"""
import numpy as np
import logging
import random
import datetime
import socket
import torch

import pickle
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from time import sleep
import os
import matplotlib.pyplot as plt
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def new_log(logdir, filename):
    """Defines logging format.
    """
    filename = os.path.join(logdir,
                            datetime.datetime.now().strftime(
                                "log_%Y-%m-%d-%H-%M-%S_" + socket.gethostname() + "_" + filename + ".log"))
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        format="%(asctime)s - %(name)s - %(message)s",
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def haversine(input_coords,
              pred_coords):
    """ Calculate the haversine distances between input_coords and pred_coords.

    Args:
        input_coords, pred_coords: Tensors of size (...,N), with (...,0) and (...,1) are
        the latitude and longitude in radians.

    Returns:
        The havesine distances between
    """
    R = 6371
    lat_errors = pred_coords[..., 0] - input_coords[..., 0]
    lon_errors = pred_coords[..., 1] - input_coords[..., 1]
    a = torch.sin(lat_errors / 2) ** 2 \
        + torch.cos(input_coords[:, :, 0]) * torch.cos(pred_coords[:, :, 0]) * torch.sin(lon_errors / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    d = R * c
    return d


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.

    Args:
        att_logits: a Tensor of shape (bachsize, data_size).
        att_idxs: a Tensor of shape (bachsize, 1), indicates
            the current idxs.
        r_vicinity: number of values to be kept.
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0], 1)
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity / 2] = -float('Inf')
    return out

def get_type():
    with open('data/ct_dma3/ct_dma3_train.pkl', 'rb') as f:
        with open('data/ct_dma3/ct_dma3_test.pkl', 'rb') as f2:
            with open('data/ct_dma3/ct_dma3_valid.pkl', 'rb') as f3:
                obj = pickle.load(f)
                obj2 = pickle.load(f2)
                obj3 = pickle.load(f3)
                tmp = dict()
                for i in range(len(obj)):
                    line = obj[i]
                    tmp[line['type']] = 1
                for i in range(len(obj2)):
                    line = obj2[i]
                    tmp[line['type']] = 1
                for i in range(len(obj3)):
                    line = obj3[i]
                    tmp[line['type']] = 1
                print(1)


def generate_similarity(model):
    lat_emb = model.state_dict().get('lat_emb.weight')
    lon_emb = model.state_dict().get('lon_emb.weight')
    lat_emb = lat_emb/lat_emb.norm(dim=-1, keepdim=True)
    lon_emb = lon_emb/lon_emb.norm(dim=-1, keepdim=True)
    # lat_sum = lat_emb.sum(dim=-1)
    # lon_sum = lon_emb.sum(dim=-1)
    lat_sim = lat_emb@lat_emb.t()
    lon_sim = lon_emb@lon_emb.t()
    lat_sim_np = lat_sim.detach().cpu().numpy()
    lon_sim_np = lon_sim.detach().cpu().numpy()

    # # 使用 imshow 展示图像，并指定颜色映射表
    # plt.imshow(lat_sim_np, cmap='viridis')
    # # 添加颜色条
    # plt.colorbar()
    # plt.show()
    # plt.close()
    #
    # # 使用 imshow 展示图像，并指定颜色映射表
    # plt.imshow(lon_sim_np, cmap='viridis')
    # # 添加颜色条
    # plt.colorbar()
    # plt.show()
    # plt.close()
    return lat_sim_np, lon_sim_np
if __name__ == '__main__':
    get_type()