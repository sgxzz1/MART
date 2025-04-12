import pickle
from tqdm import tqdm
import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import datasets
from models import Cosais
from config import Config
import utils
import matplotlib

import matplotlib.pyplot as plt
import trainers

utils.set_seed(42)
# predicted time
hours = 4
# configuration
cf = Config('ct_dma')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
if __name__ == '__main__':
    device = cf.device
    init_seqlen = cf.init_seqlen
    # if cf.pretrain:
    #     sample = False
    #     cf.n_samples = 1
    # else:
    #     sample = True
    # cf.n_samples = 1
    ## Logging
    # ===============================
    if not os.path.isdir(cf.save_dir):
        os.makedirs(cf.save_dir)
        print('======= Create directory to store trained models: ' + cf.save_dir)
    else:
        print('======= Directory to store trained models: ' + cf.save_dir)
    utils.new_log(cf.save_dir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.train_dataset, cf.valid_dataset, cf.test_dataset]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = filename
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        aisdatasets[phase] = datasets(Data[phase], cf.lon_size, cf.lat_size, cf.max_seqlen + 1)
        if phase == "test":
            shuffle = False
            aisdls[phase] = DataLoader(aisdatasets[phase],
                                       batch_size=32,
                                       shuffle=shuffle)
        else:
            shuffle = True
            aisdls[phase] = DataLoader(aisdatasets[phase],
                                       batch_size=cf.batch_size,
                                       shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen
    model = Cosais(cf)
    model = model.to(cf.device)
    ## Model
    # ===============================
    # world_size = torch.cuda.device_count()
    world_size = 2
    if cf.continue_train:
        print('continue_train')
        model.load_state_dict(torch.load(cf.ckpt_path))
    # world_size = 1
    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.save_dir, device=cf.device, aisdls=aisdls,
        INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        if cf.distributed:
            torch.multiprocessing.spawn(trainer.train_distributed, args=(world_size,), nprocs=world_size, join=True)
        else:
            trainer.train()
    ## Evaluation
    # ===============================
    # Load the best model
    model = model.to(cf.device)
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([model.lat_range, model.lon_range, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, model.lon_min, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * hours

    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        for it, (seqs, grid_indexes, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            seqs = seqs.to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            for i_sample in range(cf.n_samples):
                # if cf.pretrain:
                #     preds = trainers_nei.sample_pretrain(model,
                #                                          seqs=seqs,
                #                                          steps=max_seqlen - init_seqlen,
                #                                          sample=sample,
                #                                          r_vicinity=cf.r_vicinity,
                #                                          top_k=cf.top_k)
                # else:
                preds = trainers.sample(model,
                                            seqs=seqs_init,
                                            steps=max_seqlen - init_seqlen,
                                            sample=True,
                                            r_vicinity=cf.r_vicinity,
                                            top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
            if cf.error_graph == 'global':
                l_min_errors.append(error_ens.min(dim=-1).values)
                l_mean_errors.append(error_ens.mean(dim=-1))
            else:
                error_sum = error_ens.sum(1)
                index_list = []
                res = torch.zeros((error_ens.shape[0], error_ens.shape[1]))
                for i in range(error_sum.shape[0]):
                    tmp = error_sum[i]
                    index = tmp.argmin()
                    index_list.append(index)
                for i in range(error_ens.shape[0]):
                    res[i] = error_ens[i, :, index_list[i]]
                l_min_errors.append(res)
            l_masks.append(masks[:, cf.init_seqlen:])

    l_min = [x for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0).to(device) * m_masks
    a = m_masks.sum(dim=0)
    b = min_errors.sum(dim=0)
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    ## Plot
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors) + 1) / 6
    pred_errors = np.insert(pred_errors, 0, 0, axis=0)
    plt.plot(v_times, pred_errors)

    for i in range(1, hours + 1):
        timestep = i * 6
        plt.plot(i, pred_errors[timestep], "o")
        plt.plot([i, i], [0, pred_errors[timestep]], "r")
        plt.plot([0, i], [pred_errors[timestep], pred_errors[timestep]], "r")
        plt.text(i + 0.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    # plt.xlim([0, 12])
    # plt.ylim([0, 20])
    plt.xlim([0, 15])
    plt.ylim([0, 30])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(os.path.join(cf.save_dir,
                             f"prediction_error_init_{cf.init_seqlen}_{cf.error_graph}_{hours}h_samples_{cf.n_samples}.png"))
    np.save(os.path.join(cf.save_dir,
                         f"prediction_error_init_{cf.init_seqlen}_{cf.error_graph}_{hours}h_samples_{cf.n_samples}.npy"),
            pred_errors)
