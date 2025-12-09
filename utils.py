import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from scipy.spatial import distance
#from pytorch_msssim import ssim
from collections import namedtuple
loss_list = []
loss_list_val = []
def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    writer = None
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"
    # 定义学习率调度器，基于训练的进程在特定epoch时调整学习率。
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    # 在每个epoch中遍历训练数据，计算损失并进行反向传播，更新模型参数。
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        print("training process")
        print('current epoch:',epoch_no)
        avg_loss = 0
        avg_loss_denoise = 0
        avg_loss_predict = 0
        model.train()
        # tqdm用来显示进度条
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                # 梯度置零
                optimizer.zero_grad()
                # 求损失函数并反向传播
                loss = model(train_batch, epoch_no) #----------------------------------#
                loss[2].backward()
                avg_loss += loss[2].item()
                avg_loss_denoise += loss[0].item()
                avg_loss_predict += loss[1].item()
                global loss_list
                loss_list.append(avg_loss / batch_no)
                # 优化器调整参数
                optimizer.step()
                # 进度条显示
                
                it.set_postfix(
                    ordered_dict={
                        "avg_sum_loss": avg_loss / batch_no,
                        "train_denoise_loss": avg_loss_denoise/ batch_no,
                        "train_predict_loss": avg_loss_predict / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            if writer:
                writer.add_scalar('Loss/train', avg_loss / batch_no, epoch_no)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch_no)
            #更新学习率
            lr_scheduler.step()

            model.eval()
            valid_loss = 0.0
            valid_avg_loss_denoise = 0
            valid_avg_loss_predict = 0
            with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                with torch.no_grad():
                    for batch_no, test_batch in enumerate(it, start=1):

                        # 求损失函数并反向传播
                        loss = model(test_batch)
                        valid_loss += loss[2].item()
                        valid_avg_loss_denoise += loss[0].item()
                        valid_avg_loss_predict += loss[1].item()
                        global loss_list_val
                        loss_list_val.append(valid_loss / batch_no)

                        it.set_postfix(
                            ordered_dict={
                                "Valid_sum_loss": valid_loss / batch_no,
                                "Valid_denoise_loss": valid_avg_loss_denoise / batch_no,
                                "Valid_predict_loss": valid_avg_loss_predict / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

    if foldername != "":
        torch.save(model.state_dict(), output_path)

# 计算分位数损失：根据分位数q来计算损失，希望预测值大于观测值的q分位数
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
        #如果 target <= forecast，则结果是 (forecast - target) * eval_points *(1 - q)，意味着高于预测值的目标将受到更大的惩罚（如果 q 较大）。
        #如果 target > forecast，则结果是 (forecast - target) * eval_points *(-q)，意味着低于预测值的目标将受到更大的惩罚（如果 q 较小）
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    forecast = forecast.unsqueeze(1)
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, 1)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], 1)# (205,168,3), (205,3)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", my_scalers = None):
    print('testnsample=:',nsample)
    with torch.no_grad():
        model.eval()
        evalpoints_total = 0
        evalpoints_ssim = 0
        js_total = 0
        js_one_total = 0
        eval_js_total = 0
        evalpoints_one_total = 0
        ssim_value = 0
        tv_distance_total =0
        all_target = []
        all_observed_time = []
        all_generated_samples = []
        cut=5
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                sample, c_targets, observed_time = output
                sample = sample.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_targets = c_targets.permute(0, 2, 1)  # (B,L,K)
                
                
                # samples = sample[:,:,cut:-cut,:]
                # c_target = c_targets[:,cut:-cut,:]
                samples = sample
                c_target = c_targets
                print(c_target.shape)
                B, L, K = c_target.shape

                samples_median = samples.median(dim=1)

                # if my_scalers is not None:
                #     # 反归一化目标数据
                #     c_target = my_scalers.inverse_transform(c_target.cpu().numpy().reshape(-1, 1)).reshape(B, L, K)

                #     # 反归一化生成的样本数据
                #     samples_flat = samples.cpu().numpy()
                #     samples_flat = np.squeeze(samples_flat)
                #     # 使用 my_scaler 反归一化
                #     samples_flat = my_scalers.inverse_transform(samples_flat.reshape(-1, 1)).reshape(samples.shape)  # 反归一化并恢复为 (B, L)
                #     # 转回 PyTorch 张量并返回原始设备
                #     samples = torch.tensor(samples_flat).to(samples.device)

                #     #反归一化sample_median
                #     # 获取samples_median的中位数值（values部分）
                #     samples_median_values = samples_median.values.cpu().numpy()
                #     samples_median_values = np.squeeze(samples_median_values)

                #     # 对第一维的中位数值进行反归一化
                #     samples_median_values = my_scalers.inverse_transform(samples_median_values.reshape(-1, 1)).reshape(samples_median.values.shape)

                #     # 将反归一化后的中位数值转换回张量，并与原始indices一起组成新的samples_median
                #     samples_median_values = torch.tensor(samples_median_values).to(samples_median.values.device)
                #     # 重新构建samples_median，保持原来的格式
                #     samples_median = namedtuple('Median', ['values', 'indices'])(
                #         values=samples_median_values,
                #         indices=samples_median.indices
                #     )
                #反归一化目标和样本
#                if my_scalers is not None:
#                    # 反归一化目标数据
#                    c_target = np.stack([
#                        my_scalers[k].inverse_transform(c_target[:, :, k].cpu().numpy().reshape(-1, 1)).reshape(B, L)
#                        #my_scalers[k].inverse_transform(c_target[:, :, k].cpu().numpy()).reshape(B, L)
#                        for k in range(K)
#                    ], axis=2)
#
#                    # 反归一化生成的样本数据
#                    samples_flat = samples.cpu().numpy()
#                    samples_flat = np.stack([
#                        my_scalers[k].inverse_transform(samples_flat[:, :,:, k].squeeze(1).reshape(-1, 1)).reshape(B, L)
#                        #my_scalers[k].inverse_transform(samples_flat[:, :,:, k].squeeze(1)).reshape(B, L)
#                        for k in range(K)
#                    ], axis=2)
#                    # 转回 PyTorch 张量并返回原始设备
#                    samples = torch.tensor(samples_flat).to(samples.device)
#
#                    #反归一化sample_median
#                    # 获取samples_median的中位数值（values部分）
#                    samples_median_values = samples_median.values.cpu().numpy()
#                    samples_median_values = np.stack([
#                        my_scalers[k].inverse_transform(samples_median_values[:, :, k].reshape(-1, 1)).reshape(B, L)
#                        #my_scalers[k].inverse_transform(samples_median_values[:, :, k]).reshape(B, L)
#                        for k in range(K)
#                    ], axis=2)
#                    # 将反归一化后的中位数值转换回张量，并与原始indices一起组成新的samples_median
#                    samples_median_values = torch.tensor(samples_median_values).to(samples_median.values.device)
#                    # 重新构建samples_median，保持原来的格式
#                    samples_median = namedtuple('Median', ['values', 'indices'])(
#                        values=samples_median_values, 
#                        indices=samples_median.indices
#                    )

                c_target = torch.tensor(c_target).reshape(B, L, K).to('cuda:0')  # 转换为tensor并移动到GPU
                all_target.append(c_target)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                evalpoints_total += (B*K*L)



#----------------------------generated Metric------------------------------------
                # #：Metrix 1----JS Divers

                epsilon = 100
                # js_distance = distance.jensenshannon(samples_median.values.cpu().numpy().reshape(B,K*L).T + epsilon, c_target.cpu().numpy().reshape(B,K*L).T + epsilon, 2.0)
                flatten1_samp = samples_median.values.cpu().numpy().flatten()
                flatten1_targ = c_target.cpu().numpy().flatten()
                # 在此处对目标和预测值都做了归一化
                aaa1 = (flatten1_samp-flatten1_samp.min())/(flatten1_samp.max()-flatten1_samp.min())
                norm1_samp = aaa1/aaa1.sum()
                bbb1 = (flatten1_targ-flatten1_targ.min())/(flatten1_targ.max()-flatten1_targ.min())
                norm1_targ = bbb1/bbb1.sum()
                js_distance = distance.jensenshannon(norm1_samp, norm1_targ, 2.0)

                js_total += js_distance.item()
                #
                # #：Metrix 2----1-阶 JS Divers
                gen_dif_one = samples_median.values[:, 1:,:] - samples_median.values[:, :-1,:]
                tar_dif_one = c_target[:, 1:,:] - c_target[:, :-1, :]

                flatten2_samp = gen_dif_one.cpu().numpy().flatten()
                flatten2_targ = tar_dif_one.cpu().numpy().flatten()

                aaa2 = (flatten2_samp-flatten2_samp.min())/(flatten2_samp.max()-flatten2_samp.min())
                norm2_samp = aaa2/aaa2.sum()
                bbb2 = (flatten2_targ-flatten2_targ.min())/(flatten2_targ.max()-flatten2_targ.min())
                norm2_targ = bbb2/bbb2.sum()

                # js_distance_dif_one = distance.jensenshannon(gen_dif_one.cpu().numpy().reshape(B,-1).T + epsilon, tar_dif_one.cpu().numpy().reshape(B,-1).T + epsilon, 2.0)
                js_distance_dif_one = distance.jensenshannon(norm2_samp, norm2_targ, 2.0)

                js_one_total += js_distance_dif_one.item()
                eval_js_total += 1
                evalpoints_one_total += B*K

                # #：Metrix 3----TV-Distance
                tv_distance = 0
                for i in range(len(c_target)):
                    tv_distance += 0.5 * abs(samples_median.values[i] - c_target[i])
                tv_distance_res = tv_distance/L
                tv_distance_total +=tv_distance_res.sum().item()
                tv_distance_by_category = np.zeros(K)
                for i in range(len(c_target)):
                    # 对每个类别分别计算 TV 距离
                    for k in range(K):
                        # 计算当前类别的 TV 距离
                        tv_distance_k = 0.5 * abs(samples_median.values[i, :, k] - c_target[i, :, k])
                        tv_distance_by_category[k] += tv_distance_k.sum().item()
                tv_distance_avg_by_category = tv_distance_by_category / L

                


                ### #：Metrix 6----SSIM

#                width_base = 8
#                height = B // width_base
#                # target的图像化
##                norm_data_t = (c_target - c_target.min(dim=0,keepdim=True).values) / (c_target.max(dim=0,keepdim=True).values - c_target.min(dim=0,keepdim=True).values)
#                norm_data_t = (c_target - c_target.min()) / (c_target.max() - c_target.min())
##                scaled_data_t = (norm_data_t * 255).to(torch.uint8)
#                image_data_t = norm_data_t.permute(1,2,0).reshape(L, 1, height, width_base).float()
#
#                # generate的图像化
##                norm_data = (samples_median.values - samples_median.values.min(dim=0,keepdim=True).values) / (samples_median.values.max(dim=0,keepdim=True).values - samples_median.values.min(dim=0,keepdim=True).values)
#                norm_data = (samples_median.values - samples_median.values.min()) / (samples_median.values.max() - samples_median.values.min())
##                scaled_data = (norm_data * 255).to(torch.uint8)
#                image_data = norm_data.permute(1,2,0).reshape(L, 1, height, width_base).float()
#                
#                ssim_value_mp = ssim(image_data, image_data_t,data_range=1.0, win_size=3, size_average=False, nonnegative_ssim=True)
#                ssim_value += ssim_value_mp.max().item()
#                evalpoints_ssim += 1
                ssim_value = 1
                evalpoints_ssim = 1
# ----------------------------generated Metric------------------------------------



                it.set_postfix(
                    ordered_dict={
                        "js_total": js_total / eval_js_total,
                        "js_one_total": js_one_total / eval_js_total,
                        "tv_distance_1":tv_distance_avg_by_category[0] / evalpoints_one_total,
                        #"tv_distance_2":tv_distance_avg_by_category[1] / evalpoints_one_total,
                        #"tv_distance_3":tv_distance_avg_by_category[2] / evalpoints_one_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, mean_scaler, scaler
            )


            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        js_total / eval_js_total,
                        js_one_total / eval_js_total,
                        tv_distance_total / evalpoints_one_total,
                        CRPS,
                        ssim_value / evalpoints_ssim,
                    ],
                    f,
                )
                print(foldername)
                print("JS_div:", js_total / eval_js_total)
                print("JS_one_div:", js_one_total / eval_js_total)
                print("tv-distance:", tv_distance_total / evalpoints_one_total)
                print("CRPS:", CRPS)
                print("SSIM:", ssim_value / evalpoints_ssim)
