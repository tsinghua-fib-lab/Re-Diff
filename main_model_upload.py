import argparse

import numpy as np
import torch
import torch.nn as nn
from diff_models_csdi import diff_CSDI

from dataset_physio_4traffic_new import Area_nums
import matplotlib.pyplot as plt

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, dataset_size):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.mse = nn.MSELoss()
        self.mse2 = nn.MSELoss()
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        # if self.is_unconditional == False:
        #     self.emb_total_dim += 1  # for conditional mask
        
        # 嵌入层：将离散的输入映射到特定的维度表示
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["featureemb"] = self.emb_feature_dim

        # input_dim = 1 if self.is_unconditional == True else 2
        input_dim = 1 if self.is_unconditional == True else 1
        self.diffmodel = diff_CSDI(config_diff, input_dim, dataset_size)

        # diffusion models 的参数 β和α
        self.num_steps = config_diff["num_steps"]
        # 平方增长的 β
        if config_diff["schedule"] == "quad":
            self.beta = torch.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        # 线性增长的 β
        elif config_diff["schedule"] == "linear":
            self.beta = torch.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha = 1. - self.beta.to('cuda')
        self.alpha_hat = torch.cumprod(self.alpha,dim=0).to('cuda')

    # 平均损失
    def calc_loss_valid(
        self, observed_data, is_train,observed_tp, prompt_transfer, prompt
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, is_train, observed_tp, prompt_transfer, set_t=t, prompt=prompt
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
    
    #模拟从干净图像到噪声图像的逐渐扩散过程，即加噪过程（公式）
    def noise_images(self, x, t):
        x = x.to('cuda')
        t = t.to(self.alpha_hat.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x).to(self.alpha_hat.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    def calc_loss(
        self, observed_data,  is_train, observed_tp, prompt_transfer, prompt,epoch_no,set_t=-1
    ):
        B, K, L = observed_data.shape

        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        # current_alpha = self.alpha_torch[t]  # (B,1,1)
        # observed_data = torch.fft.rfft(observed_data, axis=-1)
        # noise = torch.randn_like(observed_data)
        # noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise #noisy+data
        noisy_data, noise = self.noise_images(observed_data,t)
        # noisy_data = noisy_data.to('cpu')
        # noise = noise.to('cpu')
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data)#here
        # predicted = self.diffmodel(total_input, observed_tp, t, data_ids)  # (B,K,L)
        total_input = total_input.unsqueeze(1) # (B, 1, K, L)
        predicted, predict_mean = self.diffmodel(total_input, observed_tp, t, prompt_transfer, prompt, epoch_no)  # (B,K,L)
        loss =self.mse(noise,predicted)
        loss2 =self.mse2(predict_mean,observed_data.mean(dim=-1,keepdim=True))
        loss_final = loss + 0.1* loss2 #--------------------------------------------------#### 超参
        loss_out = [loss, loss2, loss_final]
        return loss_out
    # 
    def set_input_to_diffmodel(self, noisy_data, observed_data):

        # 将嵌入扩展到与 noisy_data 形状一致
        B, K, L = noisy_data.shape
        # 将 data_embeddings 扩展到与 noisy_data 匹配的形状
        # data_embeddings = data_embeddings.unsqueeze(-1).unsqueeze(-1)  # (B, emb_feature_dim, 1)
        # data_embeddings = data_embeddings.expand(-1, -1, K, L)  # (B, emb_feature_dim, K, L)
        #如果是无条件模型，意味着模型不依赖于任何条件信息（比如标签、部分已知数据等），只需要用噪声图像作为输入
        if self.is_unconditional == True:
            # total_input = noisy_data.unsqueeze(-1)  # (B,K,L,1)
            total_input = noisy_data  # (B,K,L,1)
            # total_input =torch.cat([noisy_data, observed_data], dim=1) # (B,2,K,L)
        # 如果是有条件模型，将 observed_data 和 noisy_data 在通道维度上拼接，作为模型的条件输入。
        else:
            cond_obs = observed_data.unsqueeze(1)
            noisy_target = noisy_data.unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        #print(total_input.shape, data_embeddings.shape)
        #total_input = torch.cat([total_input, data_embeddings], dim=1)  # (B, 2+16, K, L)
        #print(f"set_input_to_diffmodel output shape: {total_input.shape}")
        return total_input

    def impute(self, observed_data,n_samples, observed_tp, prompt_transfer, prompt):
        # observed_data = observed_data[:,:,:48]
        B, K, L = observed_data.shape

        # _, _, E = observed_fft.shape

        # observed_data = observed_data.unsqueeze(1)



        imputed_samples = torch.zeros( B, n_samples,K, L).to(self.device)
        for i in range(n_samples):
            x = torch.randn_like(observed_data)


            # current_sample = torch.stack((in_one_one, in_two_two,), dim=1)  # B,C=2,K,E
            for t in reversed(range(1, self.num_steps)):

                total_input = x.unsqueeze(1)
                predicted, mean_predict = self.diffmodel(total_input, observed_tp.to(self.device), torch.tensor([t]).to(self.device), prompt_transfer.to(self.device), prompt)
                alpha = self.alpha[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                alpha_hat = self.alpha_hat[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                beta = self.beta[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                # coeff1 = 1 / self.alpha_hat[t] ** 0.5
                # coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                # current_sample = coeff1 * (x - coeff2 * predicted)

                if t > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)


                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted) + torch.sqrt(
                    beta) * noise


            record_sample = x
                # if t > 0:


            imputed_samples[:,i] = record_sample.detach()
        # plt.scatter(mean_predict.cpu(), observed_data.mean(dim=-1).cpu())
        # plt.plot([observed_data.mean(dim=-1).min().cpu(), observed_data.mean(dim=-1).max().cpu()], [observed_data.mean(dim=-1).min().cpu(), observed_data.mean(dim=-1).max().cpu()], 'r--')  # 理想的重构结果
        # plt.show()
        return imputed_samples

    def forward(self, batch, epoch_no=0, is_train=1):
        (
            observed_data,
            observed_tp,
            prompt_transfer,
            prompt
        ) = self.process_data(batch)
        if(is_train == 1):
            loss_func = self.calc_loss
            return loss_func(observed_data, is_train, observed_tp, prompt_transfer, prompt, epoch_no)
        else:
            loss_func = self.calc_loss_valid
            return loss_func(observed_data, is_train, observed_tp, prompt_transfer, prompt)
        #loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        #return loss_func(observed_data, is_train, observed_tp, prompt_transfer, prompt)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_tp,
            prompt_transfer,
            prompt
        ) = self.process_data(batch)

        with torch.no_grad():
            # cond_mask = gt_mask
            # target_mask = observed_mask - cond_mask

            # side_info = self.get_side_info(observed_tp, cond_mask)
            # 确保 min_val 和 max_val 是 GPU 张量
            samples = self.impute(observed_data, n_samples, observed_tp, prompt_transfer, prompt)
            # for i in range(len(cut_length)):  # to avoid double evaluation
            #     target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data,observed_tp


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=Area_nums()):
        super(CSDI_Physio, self).__init__(target_dim, config, device, 2016) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        prompt_transfer = batch["prompt_transfer"].to(self.device).float()
        prompt = batch['prompt']
        #data_ids = batch["idex_test"].to(self.device).int()

        observed_data = observed_data.permute(0, 2, 1)

        # cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        # for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_tp,
            prompt_transfer,
            prompt
        )
