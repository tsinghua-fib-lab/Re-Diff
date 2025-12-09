import argparse
import torch
import datetime
import json
import yaml
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import scipy.sparse as sp
from main_model_upload import CSDI_Physio
from dataset_physio_4traffic_new import get_dataloader
from utils import train, evaluate,loss_list
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def setup_init(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
init_seed = 11
setup_init(init_seed)
# 检查GPU是否可用
print(torch.cuda.is_available())
# print(torch.cuda.device_count())  # 显示可用的 GPU 数量
# for i in range(torch.cuda.device_count()):
#     print(torch.cuda.get_device_name(i))  # 打印每个 GPU 的名称
# 设置使用的GPU


# 定义参数解析器
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")  # 配置文件
parser.add_argument('--device', default='cuda:0', help='选择设备')
parser.add_argument("--seed", type=int, default=1)  # 随机种子
parser.add_argument("--testmissingratio", type=float, default=0.1)  # 测试集的缺失率

# n-fold交叉验证，用于评估模型泛化能力
parser.add_argument(
    "--nfold", type=int, default=0, help="用于5折测试 (有效值:[0-4])"
)

# 是否为无条件模型
parser.add_argument("--unconditional", default=True)

# 模型保存的文件夹
parser.add_argument("--modelfolder", type=str, default="")

# 生成样本数量
parser.add_argument("--nsample", type=int, default=1)

# 模型输入的通道数
parser.add_argument("--c_in", type=int, default=64)

# 图卷积类型
parser.add_argument("--graph_conv_type", type=str, default='graph_conv')

# 再次检查CUDA设备是否可用
print(torch.cuda.is_available())

# 解析输入参数
args = parser.parse_args()

# 设置计算设备（如果有GPU则使用CUDA，否则使用CPU）
args.device = torch.device('cuda:0')
#args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {args.device}")

# 读取配置文件
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# 更新配置中的模型参数
config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

# 打印配置内容
print(json.dumps(config, indent=4))

# 创建保存模型和结果的文件夹，使用当前时间命名
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print('模型文件夹:', foldername)
os.makedirs(foldername, exist_ok=True)

# 将配置文件保存到模型文件夹中
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# 获取数据加载器
train_loader, valid_loader, test_loader, my_scalers = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"]
)

# 初始化模型并将其放置到指定设备上
model = CSDI_Physio(config, args.device).to(args.device)

#model.load_state_dict(torch.load("./save/physio_fold0_20250317_184721/model.pth"),strict = True)

checkpoint = torch.load('predictor/checkpoint_2.tar')
model.diffmodel.predictor.load_state_dict(checkpoint, strict=True)


# 初始化 TensorBoard 写入器
# writer = SummaryWriter(log_dir=foldername + 'tensorboard_logs')
# 如果没有指定模型文件夹，进行训练；否则加载预训练模型
if args.modelfolder == "":
    print('开始训练')
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        valid_epoch_interval=5,
        foldername=foldername,
        # writer=writer
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

# 评估模型
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, my_scalers=my_scalers)
loss_list = np.array(loss_list)
print(f'setup_seed = {init_seed}')




import pickle
from scipy.spatial import distance
import joblib
from scipy.stats import entropy, wasserstein_distance

def compute_JSD(result_db, target_db, bins=100):
    result_flat = result_db.flatten()
    target_flat = target_db.flatten()

    # 2. 计算直方图的范围（保证两者相同）
    data_min = min(result_flat.min(), target_flat.min())
    data_max = max(result_flat.max(), target_flat.max())
    # 3. 计算直方图（相同范围 & 相同 bin）
    result_hist, bin_edges = np.histogram(result_flat, bins=bins, range=(data_min, data_max), density=False)
    target_hist, _ = np.histogram(target_flat, bins=bins, range=(data_min, data_max), density=False)

    # 4. 归一化直方图（转换成概率分布）
    result_prob = result_hist / result_hist.sum()
    target_prob = target_hist / target_hist.sum()

    # 5. 计算 JSD
    jsd_value = distance.jensenshannon(result_prob, target_prob)

    return jsd_value

def compute_NRMSE(result_db, target_db):

    clip_max = np.percentile(result_db, 99)
    clip_min = np.percentile(result_db, 1)
    result_db = np.clip(result_db, a_min=clip_min, a_max=clip_max)

    clip_max = np.percentile(target_db, 99)
    clip_min = np.percentile(target_db, 1)
    target_db = np.clip(target_db, a_min=clip_min, a_max=clip_max)

    # 计算 RMSE
    mse = np.mean((target_db - result_db) ** 2, axis=1) # 每个样本的 MSE
    rmse = np.sqrt(mse) # 计算 RMSE

    # 归一化
    nrmse = rmse / (target_db.max(axis=1) - target_db.min(axis=1) + 1e-10) # 避免除零

    return nrmse.mean()

#scaler_loaded = joblib.load("scalers.pkl")[0]

with open(
    foldername + "generated_outputs_nsample1.pk", "rb"
       # "./generated_outputs_nsample1(1).pk", "rb"
) as f:
    all_generated_samples,all_target,t_1,t_2,t_3=pickle.load(f)

#B,  K, _, L = all_target.shape
B,   L,K = all_target.shape
print(all_target.shape)
# # print(all_generated_samples)
# #数据读取：
# generated=all_generated_samples.median(dim=1).values.to('cpu').numpy().reshape(B,L)
generated=all_generated_samples.reshape(B*K,L)

targ=all_target.reshape(B*K,L)
generated = np.array(generated.cpu())
# generated = generated.clip(min=0)
targ = np.array(targ.cpu())
nonzero_rows = ~np.all(targ == 0, axis=1)
zero_rows = np.where(np.all(targ == 0, axis=1))[0]

# 过滤掉全零行
targ = targ[nonzero_rows]
generated = generated[nonzero_rows]
print(np.max(targ))
# targ = scaler_loaded.inverse_transform(targ.reshape(-1,1)).reshape(B-1,L)
# generated = scaler_loaded.inverse_transform(generated.reshape(-1,1)).reshape(B-1,L)


crps_idx = 4
#时间散度------------------------
aaa2 = (generated - np.min(generated,axis=-1, keepdims=True)) / (np.max(generated,axis=-1, keepdims=True) - np.min(generated,axis=-1, keepdims=True))
dddd = np.sum(aaa2,axis=-1,keepdims=True)
norm1_samp = aaa2 / dddd
# crps_gen = norm1_samp.reshape(1,-1)
crps_gen = norm1_samp

aaa1 = (targ - np.min(targ,axis=-1, keepdims=True)) / (np.max(targ,axis=-1, keepdims=True) - np.min(targ,axis=-1, keepdims=True))
ddd = np.sum(aaa1,axis=-1,keepdims=True)
norm1_targ = aaa1 / ddd
crps_real = norm1_targ
#js_distance_t = np.zeros((32, 1))
#for i in range(32):
#    js_distance_t[i] = distance.jensenshannon(norm1_samp[i],norm1_targ[i], 2.0)
#var_js_distance_temporal = np.var(js_distance_t)
#js_distance_temporal = js_distance_t.mean()
#
jsd = compute_JSD(generated,targ)

# #：Metrix 3----TV-Distance

tv_one = abs(generated - targ).mean(axis=1)
var_mae = np.var(tv_one)
mae_res = tv_one.mean()

# #：Metrix 3----RMSE

rmse_one = np.sqrt(((generated - targ) ** 2).mean(axis=1))
var_rmse = np.var(rmse_one)
rmse = rmse_one.mean()


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    fff = 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )
    return fff


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, 1)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], 1)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


CRPS_matrix = np.zeros((len(targ),1))
for i in range(len(targ)):
    if(i == 197):
        print(1)
    t = torch.tensor(crps_real[i]).reshape(1,crps_gen.shape[1])
    g = torch.tensor(crps_gen[i].reshape(1,crps_gen.shape[1]))
    CRPS_matrix[i] = calc_quantile_CRPS(
        t, g, mean_scaler=0,
        scaler=1
    )
var_crps = np.var(CRPS_matrix)
crps = CRPS_matrix.mean()
nrmse = compute_NRMSE(generated,targ)


print("JS_div_time:", jsd)
print("MAE:", mae_res)
print("RMSE:", rmse)
print("CRPS:", crps)
print("NRMSE:",nrmse)
print("epoch:",config["train"]["epochs"])