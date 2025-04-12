import torch
import os


class Config:
    def __init__(self, dataset_name):
        self.retrain = False
        self.DisLoss = True
        self.AssLoss = True
        self.precise = False
        # decide whether the q,k,v has bias
        self.bias = False
        self.error_graph = 'global'
        self.continue_train = False
        self.distributed = False
        self.init_seqlen = 18
        print('init_seqlen', self.init_seqlen)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # 一个输入点包含多少个点信息
        self.n_samples = 16
        self.target_points = 100
        self.sample_mode = "pos_vicinity"
        self.r_vicinity = 50
        self.top_k = 10
        self.max_epochs = 100
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join('data', dataset_name)
        self.train_dataset = os.path.join(self.dataset_dir, f'{self.dataset_name}_train.pkl')
        self.valid_dataset = os.path.join(self.dataset_dir, f'{self.dataset_name}_valid.pkl')
        self.test_dataset = os.path.join(self.dataset_dir, f'{self.dataset_name}_test.pkl')
        if self.dataset_name == 'ct_dma':
            self.batch_size = 32
            self.lat_size = 250
            self.lon_size = 270
            self.sog_size = 30
            self.cog_size = 72

            self.n_lat_embd = 256
            self.n_lon_embd = 256
            self.n_sog_embd = 128
            self.n_cog_embd = 128

            self.lat_min = 55.5
            self.lat_max = 58
            self.lon_min = 10.3
            self.lon_max = 13
        if self.dataset_name == 'ct_new_dma':
            self.batch_size = 32
            self.lat_size = 900
            self.lon_size = 2220
            self.sog_size = 30
            self.cog_size = 72

            self.n_lat_embd = 256
            self.n_lon_embd = 256
            self.n_sog_embd = 128
            self.n_cog_embd = 128

            self.lat_min = 51
            self.lat_max = 60
            self.lon_min = -1
            self.lon_max = 21.20
        if self.precise:
            self.lat_size = self.lat_size * 5
            self.lon_size = self.lon_size * 5
            self.r_vicinity = self.r_vicinity * 5
        self.min_seqlen = 36
        self.max_seqlen = 120
        self.full_size = self.lon_size + self.lat_size + self.sog_size + self.cog_size
        self.n_embd = self.n_lat_embd + self.n_lon_embd + self.n_sog_embd + self.n_cog_embd
        self.n_head = 8
        self.n_layer = 8
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
        # optimization parameters
        # ===================================================
        self.learning_rate = 6e-4  # 6e-4
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.weight_decay = 0.1  # only applied on matmul weights
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        self.lr_decay = True
        self.warmup_tokens = 512 * 20  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
        self.final_tokens = 260e9  # (at what point we reach 10% of original LR)
        self.num_workers = 4  # for DataLoader
        self.save_dir = os.path.join('results', f'{dataset_name}_DisLoss_{self.DisLoss}_precise_{self.precise}_bias_{self.bias}_AssLoss_{self.AssLoss}')
        self.ckpt_path = os.path.join(self.save_dir, 'model.pt')
