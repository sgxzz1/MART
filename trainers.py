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

"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import random

import math
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
# from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import functional as F

import config
import utils
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
    os.environ['MASTER_PORT'] = '12355'  # 主节点端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # 初始化进程组


# @torch.no_grad()
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=50,
           top_k=None):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time.
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    # if end != None:
    #     bs, _, _ = end.shape
    # model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed
        batch_size, seqlen, _ = seqs_cond.shape
        # logits.shape: (batch_size, seq_len, data_size)
        logits, _, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)
        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)
        # because of to_indexes minus 1
        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"),
                 aisdls=None,
                 INIT_SEQLEN=0):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        print(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        model = model.to(self.device)
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size, num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, grid_indexes, masks, seqlens, mmsis, time_starts) in pbar:
                # place data on the correct device
                # seqs shape(bs, seqs, 4)
                batch_size, max_seqlen, _ = seqs.shape
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)
                grid_indexes_input = grid_indexes.to(self.device)
                with torch.set_grad_enabled(is_train):
                    logits, loss, assloss = model(seqs, masks, grid_indexes_input, with_targets=True)
                    # losses.append(loss.item())
                    d_loss += loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:  # 10240,dma训练数据有9000多条
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    if assloss is not None:
                        # report progress
                        pbar.set_description(
                            f"epoch {epoch + 1}: loss {loss.item():.2f}. lr {lr:e} assloss {assloss.item():.2f}")
                    else:
                        pbar.set_description(
                            f"epoch {epoch + 1}: loss {loss.item():.2f}. lr {lr:e}")
            if is_train:
                if assloss is not None:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.2f}, assloss {assloss.item():.2f} , lr {lr:e}.")
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.2f}, lr {lr:e}.")
            else:
                if assloss is not None:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.2f}, assloss {assloss.item():.2f}")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.2f}")

            if not is_train:
                test_loss = d_loss / d_n
                #                 logging.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        # for epoch in range(config.max_epochs):
        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)
            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, grid_indexes, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
            n_plots = 7
            seqs = seqs.to(self.device)
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = sample(raw_model,
                           seqs_init,
                           96 - init_seqlen,
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx) / n_plots)
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color=c)
                plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
                plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)
            plt.close()
            if not good_model:
                if epoch - best_epoch >= 10:
                    print('after 10 epochs, the model can not find better model, early stop')
                    break
        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Last epoch: {epoch + 1:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)

    def train_distributed(self, rank, world_size):
        print(f"Running basic DDP example on rank {rank}.")
        self.device = f"cuda:{rank}"
        self.model = self.model.to(self.device)
        setup(rank, world_size)
        model_, config, aisdls, INIT_SEQLEN = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        model = DDP(model_, device_ids=[rank])
        raw_model = model_.module if hasattr(self.model, "module") else model_
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, sampler, data, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)

            # loader = DataLoader(data, sampler=sampler, pin_memory=True,
            #                     batch_size=config.batch_size)
            loader = DataLoader(data, sampler=sampler, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            losses = []
            if rank == 0:
                pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            else:
                pbar = enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, grid_indexes, masks, seqlens, mmsis, time_starts) in pbar:
                batch_size, max_seqlen, _ = seqs.shape
                seqs = seqs.to(self.device)
                masks = masks.to(self.device)
                grid_indexes_input = grid_indexes.to(self.device)

                # Forward the model
                with torch.set_grad_enabled(is_train):
                    # if config.pretrain:
                    #     logits, loss, cliploss = model(seqs, masks, grid_indexes=None, with_targets=True)
                    # else:
                    logits, loss, assloss = model(seqs, masks, grid_indexes_input, with_targets=True)
                    losses.append(loss.item())
                d_loss += loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    # 梯度裁剪，防止梯度爆炸，使得梯度的范数不超过最大值
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    if rank == 0:
                        pbar.set_description(
                            f"epoch {epoch + 1}: loss {loss.item():.2f}. lr {lr:e} assloss {assloss.item():.2f}")
            if rank == 0:
                if is_train:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.2f}, assloss {assloss.item():.2f} lr {lr:e}.")
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.2f}. assloss {assloss.item():.2f}")

            if not is_train:
                test_loss = float(np.mean(losses))
                local_test_loss = torch.tensor(test_loss).to(self.device)
                dist.all_reduce(local_test_loss, op=dist.ReduceOp.SUM)
                global_loss = local_test_loss.item() / dist.get_world_size()
                #                 logging.info("test loss: %f", test_loss)
                return global_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0
        data_train = self.train_dataset
        data_valid = self.test_dataset
        sampler_train = DistributedSampler(dataset=data_train, num_replicas=world_size, rank=rank, shuffle=True)
        sampler_valid = DistributedSampler(dataset=data_valid, num_replicas=world_size, rank=rank, shuffle=True)
        for epoch in range(config.max_epochs):
            sampler_train.set_epoch(epoch)
            sampler_valid.set_epoch(epoch)
            run_epoch('Training', sampler=sampler_train, data=data_train, epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', sampler=sampler_valid, data=data_valid, epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if rank == 0:
                if self.config.ckpt_path is not None and good_model:
                    best_loss = test_loss
                    best_epoch = epoch
                    self.save_checkpoint(best_epoch + 1)
            best_epoch_tensor = torch.tensor(best_epoch).to(self.device)
            best_loss_tensor = torch.tensor(best_loss).to(self.device)
            dist.broadcast(best_loss_tensor, src=0)
            dist.broadcast(best_epoch_tensor, src=0)
            best_epoch = best_epoch_tensor.item()
            best_loss = best_loss_tensor.item()
            if not good_model:
                if epoch - best_epoch >= 10:
                    print('after 10 epochs, the model can not find better model, early stop')
                    break

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            if rank == 0:
                raw_model = model_.module if hasattr(self.model, "module") else model_
                seqs, grid_indexes, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
                n_plots = 7
                init_seqlen = INIT_SEQLEN
                seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
                preds = sample(raw_model,
                               seqs_init,
                               96 - init_seqlen,
                               temperature=1.0,
                               sample=True,
                               sample_mode=self.config.sample_mode,
                               r_vicinity=self.config.r_vicinity,
                               top_k=self.config.top_k)

                img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
                plt.figure(figsize=(9, 6), dpi=150)
                cmap = plt.cm.get_cmap("jet")
                preds_np = preds.detach().cpu().numpy()
                inputs_np = seqs.detach().cpu().numpy()
                for idx in range(n_plots):
                    c = cmap(float(idx) / n_plots)
                    try:
                        seqlen = seqlens[idx].item()
                    except:
                        continue
                    plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
                    plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3,
                             color=c)
                    plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
                    plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(img_path, dpi=150)
                plt.close()

        # Final state
        if rank == 0:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            #         logging.info("saving %s", self.config.ckpt_path)
            logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
            save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
            torch.save(raw_model.state_dict(), save_path)
