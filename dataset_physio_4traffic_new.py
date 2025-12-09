import pickle
import torch
import os
import re
import numpy as np
from scipy.stats import linregress
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from utils import loss_list
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.feature_extraction.text import TfidfTransformer
# 输入文件：小区数*时间长度。
############# bs_record为(170,10)数据，bs_shanghai未读取入代码中操作



def Area_nums():
    area_channel = 1
    return area_channel

# 需要标准化的函数
def standardize_data(data):
    scaler = StandardScaler()
    # 假设数据的形状为 (samples, time_steps, features)
    # 将数据展平成二维 (samples * time_steps, features)
    #reshaped_data = data.reshape(-1, data.shape[-1])
    reshaped_data = data.reshape(-1,1)

    # 标准化数据
    scaled_data = scaler.fit_transform(reshaped_data)

    # 恢复回原形状 (samples, time_steps, features)
    scaled_data = scaled_data.reshape(data.shape)
    return scaled_data
def parse_id(seed):



    # observed_values = np.load('./dataset/traffic_range500.npy')
    observed_values = np.load('./dataset/traffic_range500_clean2k.npy')

#    observed_values_2 = np.load('./dataset/user_common.npy')
#    observed_values_3 = np.load('./dataset/prb_common.npy') #(1022, 168, 128)
#    with open('./dataset/prompt_copy.txt', 'r', encoding='utf-8') as f:
#        prompt = ['I want to generate the PRB utilization rate of the base station.' + line.strip() for line in f]
#    # #标准化


    #observed_values = observed_values.reshape(len(observed_values)*2, -1)
    # print(np.max(observed_values))
    percentile_95 = np.percentile(observed_values, 95)
    observed_values = np.clip(observed_values, 0, percentile_95)
    # print(np.max(observed_values))
    #observed_values = observed_values / np.max(observed_values)

#    observed_values_2 = observed_values_2.reshape(len(observed_values_2)*2, -1)
#    # print(np.max(observed_values_2))
#    percentile_95 = np.percentile(observed_values_2, 95)
#    observed_values_2 = np.clip(observed_values_2, 0, percentile_95)
#    # print(np.max(observed_values_2))
#    #observed_values_2 = observed_values_2 / np.max(observed_values_2)
#
#    prb = observed_values_3.reshape(len(observed_values_3)*2, -1)
#    percentile_95 = np.percentile(prb, 99)
#    prb = np.clip(prb, 0, percentile_95)
    mean_traffic = [np.mean(o) for o in observed_values]

    observed_values = np.expand_dims(observed_values, axis=-1)  # 必要




    # prompt = np.load("./prompt/POI_nearest_2_base_0.0025.npy")


    # prompt = np.load("./prompt/POI_range500.npy")
    prompt = np.load("./dataset/POI_range500_clean2k.npy")
    min_vals = prompt.min(axis=0)  # 在 B 维度上求最小值
    max_vals = prompt.max(axis=0)  # 在 B 维度上求最大值
    prompt = np.float32((prompt - min_vals) / (max_vals - min_vals))
    #

    id_fil = np.load('./dataset/BS_id_clean2k.npy')
    prompt_transfer = np.load('./dataset/aoi_coverage_matrix.npy')[id_fil]


    return observed_values, prompt_transfer, prompt


#

# 加载数据集
class Physio_Dataset(Dataset):
    def __init__(self, eval_length, use_index_list=None, seed=0, scalers=None):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        path = (
            "./data/traffic_volumn" + "_seed" + str(seed) + ".pk"
        )

        if os.path.isfile(path) == False or os.path.isfile(path) == True :  # if dataset file is none, create
            self.observed_values, self.prompt_transfer, self.prompt = np.array(parse_id(seed)[0]), np.array(parse_id(seed)[1]), parse_id(seed)[2]
            with open(path, "wb") as f:
                pickle.dump(
                    self.observed_values, f
                )
        else:  # load dataset file
            with open(path, "rb") as f:
                self.observed_values = pickle.load(
                    f
                )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        # 如果提供了 scalers，则应用于指定类别的对数据进行应用
        self.scalers = scalers

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        observed_data = self.observed_values[index]
        prompt_transfer = self.prompt_transfer[index]
        index = np.atleast_1d(index)
        prompt = [self.prompt[i] for i in index]
        s = {
            "observed_data": observed_data,
            "timepoints": np.arange(self.eval_length),
            "prompt_transfer": prompt_transfer,
            'prompt': prompt
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16):
    eval_length = 168  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dataset = Physio_Dataset(eval_length, seed=seed)
    #all_data = dataset[:]['observed_data']
    _, L, K = dataset.observed_values.shape
    
    # shuffle 于 parse_id() 函数中，防止索引超出范围
    np.random.seed(seed)
    indlist = np.arange(len(dataset))
    np.random.shuffle(indlist)
    # 划分训练集、测试集
    num_train = int(len(dataset) * 0.8)
    print(len(dataset))
    train_index = indlist[:num_train]
    test_index = indlist[num_train:]

    valid_index = test_index


    train_data = dataset[train_index]  # 通过训练集获取数据
    train_values = train_data['observed_data']  # 获取训练集中的观测数据
    all_data = dataset.observed_values

    # Initialize arrays to hold transformed data
    scaled_train = np.zeros_like(all_data[train_index])
    scaled_test = np.zeros_like(all_data[test_index])
    scaled_valid = np.zeros_like(all_data[valid_index])
    scaled_data = np.zeros_like(all_data)
    scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(K)]
    # Apply the scaling transformation to each class (feature) for the training, testing, and validation data
    for k in range(K):
        class_data_train = all_data[train_index, :, k].reshape(-1, 1)  # Extract class k data from train set
        class_data = all_data[:,:,k].reshape(-1,1)
        # Apply the transformations to the data for this class (feature)
        scaled_train[:, :, k] = scalers[k].fit_transform(class_data_train).reshape(all_data[train_index,:,k].shape)
        scaled_data[:,:,k] = scalers[k].transform(class_data).reshape(all_data[:,:,k].shape)

    # 训练数据集
    train_dataset = Physio_Dataset(
        eval_length, use_index_list=train_index, seed=seed, scalers=scalers
    )
    train_dataset.observed_values = scaled_data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#    print('[',end='')
#    for num in train_index:
#        print(f'{num}, ',end='')
#    print(']')
    # 验证数据集
    valid_dataset = Physio_Dataset(
        eval_length, use_index_list=valid_index, seed=seed, scalers=scalers
    )
    valid_dataset.observed_values = scaled_data
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print('[',end='')
    for num in valid_index:
        print(f'{num}, ',end='')
    print(']')
    # 测试数据集
    test_dataset = Physio_Dataset(
        eval_length, use_index_list=test_index, seed=seed, scalers=scalers
    )
    test_dataset.observed_values = scaled_data 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, scalers
