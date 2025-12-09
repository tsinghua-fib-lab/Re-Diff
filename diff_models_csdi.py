import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
# from model_contrast_module3 import Pair_CLIP_SI
from transformers import AutoTokenizer, AutoModel
from timm.models.vision_transformer import PatchEmbed, Attention
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu" # gelu
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight) # kaiming初始化权重
    return layer

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    


class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.norm = nn.LayerNorm(input_dim)
        
        def forward(self, x):
            # x = self.norm(x)
            x = self.fc1(x)
            # x = self.drop(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

class MLPRegressor(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.emb_data0 = nn.Sequential(nn.Linear(1, hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim,hidden_dim))
        self.emb_data1 = nn.Sequential(nn.Linear(12, hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim,hidden_dim))
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出层（预测 Y）
        )

        # self.model = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        aoi0 = (x[:,:24]).sum(dim=-1,keepdim=True)
        aoi = self.emb_data0(aoi0)
        poi = self.emb_data1(x[:,-12:])
        y = self.model(aoi+poi)
        # out = self.output(y)
        return y

# 时间步嵌入：通常用于向模型传递当前的扩散步骤信息
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        # 将 embedding 注册为一个缓冲区（而不是模型参数），表示在模型训练中不会对其进行更新，但可以保存和加载
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        #生成时间序列
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        #生成用于频率编码的频率矩阵，使用指数函数进行缩放
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        # 将时间步与频率相乘，生成一个基于不同时间步和频率的嵌入表
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):  # 定义一个名为 diff_CSDI 的神经网络模型类，继承自 nn.Module
    def __init__(self, config, inputdim=1, dataset_size=1000):  # 初始化函数，config 包含模型的参数配置，inputdim 为输入数据的维度，默认为 1
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.channels = config["channels"]  # 从配置中读取通道数

        # 定义扩散过程中的时间步嵌入模块，输入为时间步数，输出为扩散嵌入
        # 通过嵌入时间步，模型能够意识到当前数据处于扩散的哪个阶段，并根据不同的时间步对数据应用不同的处理方式，从而实现对数据的逐步去噪和重建。
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],  # 扩散过程中的步数
            embedding_dim=config["diffusion_embedding_dim"],  # 扩散嵌入的维度
        )
        # feature 嵌入层
        self.dataset_size = dataset_size
        self.emb_feature_dim = config["featureemb"]
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=self.emb_feature_dim, kernel_size=3, padding=1)
        self.mlp_transfer = MLP(input_dim=12, hidden_dim=self.channels*2, output_dim=self.channels)
        self.mlp_emb = MLP(input_dim=1, hidden_dim=self.channels*2, output_dim=self.channels)
        self.predictor = MLPRegressor()
        # for param in self.predictor.parameters():
        #     param.requires_grad = False
        #------------------------------------------------------------#

        self.attention = nn.MultiheadAttention(embed_dim=128,num_heads=config['nheads'])
        #------------------------------------------------------------#
        self.ln_time = nn.LayerNorm(12)

        self.ln_side = nn.LayerNorm(128)
        # 输入数据的卷积投影层，将输入数据维度投影到模型的通道数
        self.input_projection = Conv1d_with_init(in_channels=1, out_channels = self.channels, kernel_size=1)
        self.mean_projection = Conv1d_with_init(in_channels=1, out_channels=self.channels, kernel_size=1)
        self.prompt_projection = Conv1d_with_init(in_channels=1, out_channels=168, kernel_size=1)
        # 中间输出的投影层
        self.output_projection1 = Conv1d_with_init(self.channels, 2*self.channels, 1)
        # 最终输出的投影层，投影到 1 维（假设是单通道输出）
        self.output_projection2 = Conv1d_with_init(2*self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)  # 将最后一层的权重初始化为全零
        # 定义残差块的列表，每一层都是一个 ResidualBlock
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],  # 旁路信息的维度
                    channels=self.channels,  # 模型的通道数
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],  # 扩散嵌入的维度
                    nheads=config["nheads"],  # 多头注意力机制的头数
                )
                for _ in range(config["layers"])  # 创建指定数量的残差块
            ]
        )
        


    # 定义时间嵌入函数，将时间步转换为一个向量表示
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).cuda()  # 初始化位置编码张量，全为 0，并放到 GPU 上
        position = pos.unsqueeze(2)  # 增加一个维度，便于后续操作
        div_term = 1 / torch.pow(  # 计算分母，用于位置编码的缩放
            10000.0, torch.arange(0, d_model, 2).cuda() / d_model  # 按照 d_model 计算缩放因子
        )
        # div_term = div_term.to('cpu')
        pe[:, :, 0::2] = torch.sin(position * div_term)  # 偶数索引使用正弦函数编码
        pe[:, :, 1::2] = torch.cos(position * div_term)  # 奇数索引使用余弦函数编码
        return pe  # 返回生成的时间嵌入
######### 前向传播函数，输入 x 和扩散时间步 diffusion_step，通过时间嵌入、输入投影、残差块和 skip 连接
    def forward(self, x, observed_tp,  diffusion_step, vector_aoi, vector_poi, epoch_no=0):

        vector_poi = vector_poi[0].to("cuda:0").float() # (B,1536)
        x = x.to('cuda:0') #(B,C=1,K=2,L=168)
        B, _, K, L = x.shape  # Batch size, channels, feature count, time steps

        vector_aoi = vector_aoi.to("cuda:0")
        x = self.input_projection(x.reshape(B,-1,K*L))  # Shape: (B, channels, K*L) -> (B, out_channels, K*L)
        x = F.relu(x)
        B, C, _ = x.shape  # Batch size, channels, feature count, time steps
        x = x.reshape(B, C, K, L)
        #mean_predict = self.predictor(torch.cat([vector_aoi, vector_poi], dim=1)).unsqueeze(1).expand(B, L, 1).detach()
        if epoch_no < 30:
            mean_predict = self.predictor(torch.cat([vector_aoi, vector_poi], dim=1)).unsqueeze(1).expand(B, L, 1)
        else:
            mean_predict = self.predictor(torch.cat([vector_aoi, vector_poi], dim=1)).unsqueeze(1).expand(B, L, 1).detach()
        #mean_predict = self.predictor(torch.cat([vector_aoi, vector_poi], dim=1)).unsqueeze(1).expand(B, L, 1)
        aoi_emb_mean = self.mlp_emb(mean_predict)  # (B, C=128,K=1, L=168)

        vector_poi = vector_poi.unsqueeze(1).expand(B, L, 12)  # B,L,12
        poi_emb = self.mlp_transfer(vector_poi)

        # 对L做注意

        time_embed = self.time_embedding(observed_tp,self.channels)  # 生成 (B, L, 128) 的时间嵌入
        diffusion_emb = self.diffusion_embedding(diffusion_step)  # Shape: (B, diffusion_embedding_dim)
        
        # Prepare skip connections
        skip = []
        
        for layer in self.residual_layers:
            x, skip_connection = layer(x, time_embed, aoi_emb_mean, diffusion_emb, poi_emb)
            # x, skip_connection = layer(x, side_info, diffusion_emb)
            skip.append(skip_connection)

        # Aggregate skip connections
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        _,out_C,_,_ = x.shape

        x = x.reshape(B, out_C, K*L)
        x = self.output_projection1(x)  # Shape: (B, channels, K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # Shape: (B, 1, K*L)
        x = x.reshape(B,K,L)


        return x, mean_predict


# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        # 将扩散嵌入（时间步嵌入）投影到与输入通道一致的维度
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection0 = Conv1d_with_init(2*channels, channels, 1)
        # 对旁路信息 cond_info 进行投影，输出的通道数为 2 倍的 channels，用于后续门控机制
        self.cond_projection = Conv1d_with_init(2 * side_dim, 4 *channels, 1)
        
        # 对中间结果进行投影，输出通道数为 2 倍的 channels，用于后续的门控机制
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        # 对最终的输出进行投影，输出通道数为 2 倍的 channels，用于生成 residual 和 skip 连接
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # 用于处理时间维度上的注意力层，多头注意力机制
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.fuse = Conv1d_with_init(2 * channels, channels, 1)
        
        # 交叉注意力层：对 K 维度建模
        self.cross_attention0 = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        self.cross_attention1 = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        self.cross_attention3 = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        #self.cross_attention4 = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        self.norm0 = nn.LayerNorm(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        #self.norm4 = nn.LayerNorm(channels)
        self.ln_y = nn.LayerNorm(channels)
        #self.alpha00 = nn.Parameter(torch.randn(128))
        self.alpha0 = nn.Parameter(torch.randn(128))
        self.alpha1 = nn.Parameter(torch.randn(128))
        self.alpha2 = nn.Parameter(torch.randn(128))
        self.alpha3 = nn.Parameter(torch.randn(128))
        #self.alpha4 = nn.Parameter(torch.ones(1))
        # # 通道注意力层

        
    # 处理时间维度上的注意力
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape # K特征维度，L时间步数
        if L == 1:
            return y
        # 重塑张量使其适应时间维度的处理，permute函数用于调整维度顺序
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        # 将 y 转置为 (L, B * K, channel)，即时间步 L 现在是第一个维度。
        # 这样做是为了让时间步成为主要处理维度，对多头自注意力层的计算是必要的
        # 将y输入到多头自注意力层：计算多纬度注意力，捕捉到序列中的不同依赖关系
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # 恢复张量的原始形状
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def fourier_filter_top_x_to_x_plus_4(self, tensor, k):
        """
        对输入 tensor 在最后一个维度进行傅里叶变换，保留频域中从第 x 大到第 (x+4) 大的分量，并进行逆变换
        :param tensor: 输入 tensor，形状为 (B, 128, 168)
        :param x: 选择的频域分量排名（从第 x 大到第 x+4 大）
        :return: 处理后的 tensor，形状仍为 (B, 128, 168)
        """
        # 对最后一维进行傅里叶变换
        fft_result = torch.fft.rfft(tensor, dim=-1)

        # 获取频域的幅值
        magnitude = torch.abs(fft_result)

        # 选出最大的 (x+4) 个分量
        topk_values, topk_indices = torch.topk(magnitude, k + 4, dim=-1)

        # 选出排名在 x 到 x+4 之间的索引
        selected_indices = topk_indices[..., k:k + 4]

        # 构造掩码，初始化为全 0
        mask = torch.zeros_like(fft_result, dtype=torch.bool)

        # 选出第 x 到 x+4 大的频率分量
        mask.scatter_(-1, selected_indices, True)

        # 仅保留选定的分量，其余置零
        filtered_fft = fft_result * mask

        # 进行逆傅里叶变换
        filtered_tensor = torch.fft.irfft(filtered_fft, dim=-1) #

        return filtered_tensor

    # 前向传播，处理输入 x、旁路信息 cond_info 和扩散嵌入 diffusion_emb
    def forward(self, x, timeemb, aoi_emb, diffusion_emb, poi_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape  # 保存输入形状以便后续重塑
        x = x.reshape(B, channel, K * L)

        # 将扩散嵌入投影到与输入通道一致的维度并添加时间嵌入
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B, channel, 1)
        y = x + diffusion_emb  # 将时间步嵌入与输入叠加

        # 时间维度上的多头注意力处理
        y = self.forward_time(y, base_shape)
        y = y + timeemb.permute(0,2,1)

        # embeddings.permute(0, 2, 1)
        # total_emb = torch.cat((embeddings, prompt_transfer),dim = -1)
        # total_emb = self.fuse(total_emb.permute(0,2,1)).permute(0,2,1)

        # 写文章时写成分m个
        filtered_tensor_0 = self.fourier_filter_top_x_to_x_plus_4(x, k = 0)
        filtered_tensor_1 = self.fourier_filter_top_x_to_x_plus_4(x, k=4)
        filtered_tensor_2 = self.fourier_filter_top_x_to_x_plus_4(x, k=8)
        filtered_tensor_3 = self.fourier_filter_top_x_to_x_plus_4(x, k=12)
        #filtered_tensor_4 = self.fourier_filter_top_x_to_x_plus_4(x, k=16)
        residual = x - filtered_tensor_0 - filtered_tensor_1 - filtered_tensor_2 - filtered_tensor_3
        # cond_info = self.cond_projection0(cond_info.permute(0,2,1))
        cond_info = poi_emb + timeemb + aoi_emb
        #cond_info = timeemb
        t_c0 = self.cross_attention0(self.norm0(filtered_tensor_0.permute(0,2,1)),cond_info, cond_info, need_weights=False)[0] +filtered_tensor_0.permute(0,2,1)
        #print(t_c0.shape)
        t_c1 = self.cross_attention1(self.norm1(filtered_tensor_1.permute(0, 2, 1)), cond_info,cond_info, need_weights=False)[0] +filtered_tensor_1.permute(0,2,1)
        t_c2 = self.cross_attention2(self.norm2(filtered_tensor_2.permute(0, 2, 1)), cond_info,cond_info, need_weights=False)[0] + filtered_tensor_2.permute(0,2,1)
        t_c3 = self.cross_attention3(self.norm3(filtered_tensor_3.permute(0, 2, 1)), cond_info,cond_info, need_weights=False)[0] + filtered_tensor_3.permute(0,2,1)
        #t_c4 = self.cross_attention4(self.norm4(filtered_tensor_4.permute(0, 2, 1)), cond_info,cond_info, need_weights=False)[0] + filtered_tensor_4.permute(0,2,1)



        # 特征维度上的多头注意力处理
        # 将旁路信息与中间结果叠加
        #y =residual +t_c0.permute(0,2,1)+t_c1.permute(0,2,1)+t_c2.permute(0,2,1)+t_c3.permute(0,2,1) + y #+ torch.relu(self.alpha4) * t_c4.permute(0,2,1)
        y =residual +torch.relu(self.alpha0).view(1, -1, 1) * t_c0.permute(0,2,1)+torch.relu(self.alpha1).view(1, -1, 1) *t_c1.permute(0,2,1)+torch.relu(self.alpha2).view(1, -1, 1) *t_c2.permute(0,2,1)+torch.relu(self.alpha3).view(1, -1, 1) *t_c3.permute(0,2,1) + y #+ torch.relu(self.alpha4) * t_c4.permute(0,2,1)
        #y = self.ln_y(y.permute(0,2,1)).permute(0,2,1)
        # y = y+mean_x_cross.permute(0,2,1)
        y = self.mid_projection(y)
        # 使用门控机制（GLU）：通过 torch.chunk 将 y 拆分为 gate 和 filter
        gate, filter = torch.chunk(y, 2, dim=1)
        # 通过 sigmoid 和 tanh 激活函数对 gate 和 filter 进行门控处理
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, channel, K * L)

        # 输出投影，将门控后的结果投影回 channels
        y = self.output_projection(y)

        # 将 y 再次拆分为 residual 和 skip 连接
        residual, skip = torch.chunk(y, 2, dim=1)
        
        # 恢复 x 的形状
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        # 返回残差连接（x + residual）和跳跃连接 skip
        return (x + residual) / math.sqrt(2.0), skip