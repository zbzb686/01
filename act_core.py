import torch  # 用于创建模拟CNN输出的特征张量，本轮只在CPU运行。
from torch import nn  # 导入神经网络模块，包括模块基类和卷积层。

class ImageTokenProjector(nn.Module):  # 定义一个可复用模块，将CNN特征图转换成图像tokens。
    def __init__(self, input_channels, token_dim):  # 接收输入通道数，以及希望每个token具有的特征数量。
        super().__init__()  # 初始化父类，让PyTorch能够管理本模块的层和参数。
        self.proj = nn.Conv2d(  # 创建可学习的1×1通道投影层，各空间位置共用其权重。
            in_channels=input_channels,  # 填表示输入通道数的变量名，让模块适用于不同CNN。
            out_channels=token_dim,  # 填表示目标特征长度的变量名，让输出通道数等于token长度。
            kernel_size=1,  # 填卷积核大小，使它逐位置转换通道，不混合相邻位置。
        )  # 完成投影层的创建，权重此时由PyTorch初始化。

    def forward(self, features):  # 定义输入特征图经过这个模块时的计算流程。
        projected = self.proj(features)  # 将形状[B,C,H,W]转换为[B,D,H,W]，本例保持空间尺寸。
        flat = projected.flatten(start_dim=2)  # 将高度和宽度合并，得到[B,D,H×W]。
        tokens = flat.transpose(1, 2)  # 交换通道与位置两个维度，得到[B,H×W,D]。
        return tokens  # 返回图像tokens，后续再接入Transformer。

class StateTokenProjector(nn.Module):
    def __init__(self,state_dim,token_dim):
        super().__init__()
        self.proj=nn.Linear(in_features=state_dim,out_features=token_dim)
    def forward(self, state):
        projected=self.proj(state)
        tokens=projected.unsqueeze(1)
        return tokens

def build_observation_tokens(image_tokens, state_tokens):#拼接图像tokens和状态tokens，得到最终的观测tokens
    tokens=torch.cat([state_tokens, image_tokens], dim=1)
    return tokens

def weight_read(weights,value):
    return weights@value

def scaled_dot_product_attention(q, k, v):#计算缩放点积注意力，返回加权后的输出和注意力权重。
    weights=torch.softmax(q@k.transpose(-1,-2)/q.shape[-1]**0.5,dim=-1)
    outputs=weight_read(weights,v)
    return outputs,weights

class SingleHeadAttention(nn.Module):#单头注意力模块，包含线性投影和缩放点积注意力。
    def __init__(self, token_dim):
        super().__init__()
        self.q_proj = nn.Linear(in_features=token_dim, out_features=token_dim)
        self.k_proj=nn.Linear(in_features=token_dim, out_features=token_dim)
        self.v_proj=nn.Linear(in_features=token_dim, out_features=token_dim)
    def forward(self,tokens):
        q=self.q_proj(tokens)
        k=self.k_proj(tokens)
        v=self.v_proj(tokens)
        outputs,weights=scaled_dot_product_attention(q,k,v)
        return outputs,weights

class MultiHeadAttention(nn.Module):#多头注意力模块，包含多个单头注意力和线性投影。
    def __init__(self,token_dim,num_heads):
        super().__init__()
        self.q_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
        self.k_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
        self.v_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
        self.num_heads=num_heads
        self.head_dim=token_dim//num_heads
        self.out_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
    def forward(self,tokens,position=None):
        qk_tokens = tokens if position is None else tokens + position  # 未传位置时使用原内容；传入位置时，为Q/K准备内容与位置之和。
        batch_size,seq_len,token_dim=tuple(tokens.shape)
        q=self.q_proj(qk_tokens).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(qk_tokens).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(tokens).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        outputs,weights=scaled_dot_product_attention(q,k,v)
        outputs=outputs.transpose(1,2).reshape(batch_size,seq_len,token_dim)
        outputs=self.out_proj(outputs)
        return outputs,weights

class AttentionResidualBlock(nn.Module):
    def __init__(self,token_dim,num_heads):
        super().__init__()
        self.LayerNorm=nn.LayerNorm(token_dim)
        self.multih_head_attention=MultiHeadAttention(token_dim,num_heads)
    def forward(self,tokens,position=None):
        outputs, weights = self.multih_head_attention(tokens, position=position)  # 将本层收到的位置向量继续传给多头注意力。
        outputs=tokens+outputs
        outputs=self.LayerNorm(outputs)
        return outputs,weights

class FeedForward(nn.Module):
    def __init__(self,token_dim,hidden_dim):
        super().__init__()
        self.fcn_1=nn.Linear(in_features=token_dim,out_features=hidden_dim)
        self.relu=nn.ReLU()
        self.fcn_2=nn.Linear(in_features=hidden_dim,out_features=token_dim)
    def forward(self,tokens):
        z_1=self.relu(self.fcn_1(tokens))
        outputs=self.fcn_2(z_1)
        return outputs
    
class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_tokens, token_dim):
        super().__init__()
        self.embedding=nn.Embedding( max_tokens,token_dim)
    def forward(self,tokens):
        _,N,_=tuple(tokens.shape)
        indices = torch.arange(N, device=tokens.device)  # 生成0到N-1的位置编号，并与输入处于同一设备；当前CPU也适用。
        outputs=self.embedding(indices).unsqueeze(0) 
        return outputs

class TransformerEncoderBlock(nn.Module):
    def __init__(self,Token_dim, Num_heads, Hidden_dim):
        super().__init__()
        self.attentionResidualBlock=AttentionResidualBlock(token_dim=Token_dim,num_heads=Num_heads)
        self.FFN=FeedForward(token_dim=Token_dim,hidden_dim=Hidden_dim)
        self.LayerNorm=nn.LayerNorm(Token_dim)   
    def forward(self,tokens,position=None):
        y, weights = self.attentionResidualBlock(tokens, position=position)  # 将位置向量传给注意力残差块，原始tokens仍作为内容输入。
        outputs=self.LayerNorm(self.FFN(y)+y)
        return outputs,weights

class TransformerEncoder(nn.Module):
    def __init__(self, token_dim,num_heads,hidden_dim,num_layers):
            super().__init__()
            self.layers=nn.ModuleList()
            for _ in range(num_layers):
                layer=TransformerEncoderBlock(Token_dim=token_dim,Num_heads=num_heads,Hidden_dim=hidden_dim)
                self.layers.append(layer)
    def forward(self,tokens,position=None):
            weights=[]
            tokens_layer=tokens
            for layer in self.layers:
                tokens_layer,weight=layer(tokens_layer,position=position)
                weights.append(weight)
            return tokens_layer,weights

class CrossAttention(nn.Module):
    def __init__(self,token_dim,num_heads):
        super().__init__()
        self.q_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
        self.k_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
        self.v_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
        self.num_heads=num_heads
        self.head_dim=token_dim//num_heads
        self.out_proj=nn.Linear(in_features=token_dim,out_features=token_dim)
    def forward(self,action_tokens,observation_tokens,act_position=None,obs_position=None):
        qk_1tokens = action_tokens if act_position is None else action_tokens + act_position  # 未传位置时使用原内容；传入位置时，为Q/K准备内容与位置之和。
        qk_2tokens = observation_tokens if obs_position is None else observation_tokens + obs_position  # 未传位置时使用原内容；传入位置时，为Q/K准备内容与位置之和。
        batch_size,act_seq_len,token_dim=tuple(action_tokens.shape)#不同token的维度一样batch一样
        _,obs_seq_len,_=tuple(observation_tokens.shape)
        q=self.q_proj(qk_1tokens).reshape(batch_size,act_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(qk_2tokens).reshape(batch_size,obs_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(observation_tokens).reshape(batch_size,obs_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        outputs,weights=scaled_dot_product_attention(q,k,v)
        outputs=outputs.transpose(1,2).reshape(batch_size,act_seq_len,token_dim)
        outputs=self.out_proj(outputs)
        return outputs,weights

class TransformerDecoderBlock(nn.Module):
    def __init__(self,token_dim,num_heads,hidden_dim):
            super().__init__()
            self.self_attention=AttentionResidualBlock(token_dim,num_heads)
            self.cross_attention=CrossAttention(token_dim,num_heads)
            self.cross_layernorm=nn.LayerNorm(token_dim)
            self.ffn_layernorm=nn.LayerNorm(token_dim)
            self.ffn=FeedForward(token_dim,hidden_dim)
    def forward(self,act_tokens,obs_tokens,act_position=None,obs_position=None):
        y,self_weights=self.self_attention(act_tokens,act_position)
        cross_output,cross_weights=self.cross_attention(y,obs_tokens,act_position,obs_position)
        z=self.cross_layernorm(y+cross_output)
        outputs=self.ffn_layernorm(z+self.ffn(z))
        return outputs,self_weights,cross_weights
                
class ActionHead(nn.Module):
    def __init__(self, token_dim, action_dim):
        super().__init__()
        self.act_proj=nn.Linear(token_dim,action_dim)
    def forward(self,decode_tokens):
        outputs=self.act_proj(decode_tokens)
        return outputs

class TransformerDecoder(nn.Module):
    def __init__(self, token_dim,num_heads,hidden_dim,num_layers):
            super().__init__()
            self.layers=nn.ModuleList()
            for _ in range(num_layers):
                layer=TransformerDecoderBlock(token_dim,num_heads,hidden_dim)
                self.layers.append(layer)
    def forward(self,act_input,encoder_input,act_position=None,obs_position=None):
            weights=[]
            decoder_tokens_layer=act_input#解码器的输入输出迭代的是act的tokens
            encoder_input_layer=encoder_input
            for layer in self.layers:
                decoder_tokens_layer,self_weight,cross_weight=layer(decoder_tokens_layer,encoder_input_layer,act_position,obs_position)
                weights.append((self_weight,cross_weight))
            return decoder_tokens_layer,weights

def masked_l1_loss(predicted_actions, target_actions, action_is_pad):
    abs=torch.abs(predicted_actions-target_actions)
    valid_mask = (~action_is_pad).unsqueeze(-1)  # 将“是否补齐”取反为“是否有效”，并把[B,A]变成[B,A,1]。
    error_sum=(abs*valid_mask).sum()
    valid_count=valid_mask.sum() * predicted_actions.shape[-1] 
    loss = error_sum / valid_count.clamp_min(1)  # 计算平均误差；分母至少为1，避免全补齐时除以零。
    return loss

class VisualBackbone(nn.Module):  # 把原始RGB图像逐级缩小成低分辨率特征图，取代此前用手工模拟的CNN输出。
    downsample_factor = 16  # 四层步长为2的卷积，每层把高宽减半，累计缩小16倍：96→48→24→12→6。

    def __init__(self, image_channels, out_channels):  # 接收输入图像的通道数，以及希望输出的特征通道数。
        super().__init__()  # 初始化父类，让PyTorch统一管理下面的卷积权重。
        self.features = nn.Sequential(  # 按顺序串联四组“卷积+激活”。
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),  # 第1层：通道升到32，空间高宽各减半。
            nn.ReLU(),  # 非线性激活，让网络能表达比线性卷积更复杂的图像模式。
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 第2层：通道升到64，空间再减半。
            nn.ReLU(),  # 再次引入非线性。
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 第3层：保持64通道，继续缩小空间尺寸。
            nn.ReLU(),  # 保持非线性。
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),  # 第4层：通道数变为后续token投影所需的特征通道数。
            nn.ReLU(),  # 输出前再激活一次，使特征非负。
        )  # 完成整个视觉特征提取器。

    def forward(self, images):  # 输入图像张量，输出缩小后的特征图。
        return self.features(images)  # 逐层计算，形状由[B,3,96,96]变为[B,out_channels,6,6]。

def reparameterize(mu, logvar):  #接收均值和对数方差，返回采样的潜变量。
    std = torch.exp(0.5 * logvar)  # 先把对数方差转换为标准差。
    eps = torch.randn_like(std)  # 生成与标准差形状相同的标准正态噪声。
    return mu + eps * std  # 按重参数化公式计算潜变量。

class VaeEncoder(nn.Module): 
    def __init__(self, state_dim,action_dim,token_dim,num_head,num_hidden,action_count,latent_dim,num_layers):
        super().__init__()
        self.cls_token=nn.Parameter(torch.randn(1, 1, token_dim)*0.02)
        self.stateproj=StateTokenProjector(state_dim,token_dim)
        self.actproj=nn.Linear(action_dim,token_dim)
        self.position=LearnedPositionEmbedding(action_count+2,token_dim)
        self.transformer=TransformerEncoder(token_dim,num_head,num_hidden,num_layers)
        self.to_latent=nn.Linear(token_dim,2*latent_dim)
    def forward(self,state,actions):
        batch_size=state.shape[0]
        cls_token=self.cls_token.expand(batch_size, -1, -1)
        state_tokens=self.stateproj(state)
        action_tokens=self.actproj(actions)
        input_tokens=torch.cat([cls_token,state_tokens,action_tokens],dim=1)
        position=self.position(input_tokens)
        encoded,_=self.transformer(input_tokens,position)
        cls_out=encoded[:,0]
        params=self.to_latent(cls_out)
        mu=params[:, :params.shape[-1] // 2]
        logvar=params[:, params.shape[-1] // 2:]
        return mu,logvar

def kl_divergence(mu, logvar):  # 接收均值和对数方差，返回这批样本与标准正态的平均KL散度。
    kl_per_sample = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1)  # 先逐维套公式，再沿latent维求和，得到每条样本一个KL。
    return kl_per_sample.mean()  # 再对批次求平均；这样KL的量级不会随batch_size变化。


class MiniACT(nn.Module):
    def __init__(self,image_channels,backbone_channels,state_dim,action_dim,token_dim,num_head,num_hidden,action_count,max_obs_token,encoder_layer,decoder_layer,latent_dim,vae_layers ):
        super().__init__()
        self.backbone=VisualBackbone(image_channels,backbone_channels)  # 先由卷积骨干把原始图像压成低分辨率特征图。
        self.image_proj=ImageTokenProjector(backbone_channels,token_dim)  # 再把特征图逐位置投影成token。
        self.state_proj=StateTokenProjector(state_dim,token_dim)  # 状态也要变成一个token参与编码。
        self.vae=VaeEncoder(state_dim,action_dim,token_dim,num_head,num_hidden,action_count,latent_dim,vae_layers)  # 只在训练时使用：从示范动作推断潜变量分布。
        self.latent_proj=nn.Linear(latent_dim,token_dim)  # 把潜变量z投影成和其他token相同的特征长度。
        self.act_position=LearnedPositionEmbedding(action_count,token_dim)  # 动作查询位置表。
        self.obs_position=LearnedPositionEmbedding(max_obs_token+1,token_dim)  # 观测位置表，+1是给最前面的latent token预留的格子。
        self.encoder=TransformerEncoder(token_dim,num_head,num_hidden,encoder_layer)  # 主编码器，输入含latent的观测序列。
        self.decoder=TransformerDecoder(token_dim,num_head,num_hidden,decoder_layer)  # 解码器，动作查询从中读取观测信息。
        self.action_head=ActionHead(token_dim,action_dim)  # 共享的动作输出头。
        self.action_count = action_count  # 保存每次预测的动作数量，供forward创建动作位置使用。
        self.latent_dim = latent_dim  # 保存潜变量维度，推理时用它创建全零z。
    def forward(self,images,state,target_actions=None):
        features=self.backbone(images)  # 先用卷积骨干把原始图像[B,3,96,96]压成特征图[B,C,6,6]。
        image_tokens=self.image_proj(features)  # 特征图逐位置投影成图像token。
        state_tokens=self.state_proj(state)  # 状态投影成单个token。
        obs_tokens=build_observation_tokens(image_tokens,state_tokens)  # 拼成观测序列，状态在前。
        if self.training and target_actions is not None:  # 训练且提供了示范动作，才能走VAE编码器。
            mu,logvar=self.vae(state,target_actions)  # 从示范动作推断潜变量分布参数。
            latent=reparameterize(mu,logvar)  # 用重参数化技巧采样，保证梯度能回到VAE编码器。
        else:  # 推理时没有示范动作，无法计算分布参数。
            mu,logvar=None,None  # 返回None，调用者据此跳过KL项。
            latent=obs_tokens.new_zeros((obs_tokens.shape[0],self.latent_dim))  # 直接把z设为全零，即取先验分布的均值。
        latent_token=self.latent_proj(latent).unsqueeze(1)  # 把[B,L]的z投影并补出token维，得到[B,1,D]。
        encoder_input=torch.cat([latent_token,obs_tokens],dim=1)  # latent放在序列最前面，与状态、图像token一起送进编码器。
        position=self.obs_position(encoder_input)  # 为整条序列（含latent）取位置向量。
        encoder_output,_=self.encoder(encoder_input,position)  # 主编码器融合latent、状态与视觉信息。
        action_tokens = encoder_output.new_zeros((encoder_output.shape[0], self.action_count, encoder_output.shape[-1]))  # 为A个待预测动作位置创建零内容特征。
        act_position=self.act_position(action_tokens)  # 取动作查询的位置编码。
        decoder_outputs,_=self.decoder(action_tokens,encoder_output,act_position,position)  # 动作查询从编码器输出读取信息；注意位置长度要与encoder_output一致。
        predicted_actions=self.action_head(decoder_outputs)  # 每个动作位置投影成二维目标坐标。
        return predicted_actions,mu,logvar  # 动作预测与潜变量分布参数一起返回，供计算L1和KL。



if __name__ == "__main__":  # 只在直接运行本文件时执行下面的演示，导入模块时不执行。
    features = torch.randn(2, 8, 3, 3)  # 模拟两张图的CNN特征输出，每张8通道、空间3×3。
    state=torch.randn(2,2)
    state_projector=StateTokenProjector(state_dim=2,token_dim=64)
    state_tokens=state_projector(state)
    projector = ImageTokenProjector(input_channels=8, token_dim=64)  # 创建把每处8个特征转换成64个特征的模块。
    image_tokens = projector(features)  # 调用模块，PyTorch会执行上面定义的forward方法。
    print("输入特征图形状：", tuple(features.shape),"输入状态特征形状：", tuple(state.shape))  # 查看输入的批次、通道、高度和宽度。
    print("输出image_tokens形状：", tuple(image_tokens.shape),"输出state_tokens形状：", tuple(state_tokens.shape))  # 查看输出的批次、位置数量和每个位置的特征长度。
    observation_tokens=build_observation_tokens(image_tokens,state_tokens)
    print("输出observation_tokens形状：", tuple(observation_tokens.shape))  # 查看输出的批次、位置数量和每个位置的特征长度。
    single_head_attention=SingleHeadAttention(token_dim=64)
    attention_output,attention_weights=single_head_attention(observation_tokens)
    print("注意力输出形状：", tuple(attention_output.shape),"注意力权重形状：", tuple(attention_weights.shape))  # 查看输出的批次、位置数量和每个位置的特征长度。
    multi_head_attention=MultiHeadAttention(token_dim=64,num_heads=4)
    multi_attention_output,multi_attention_weights=multi_head_attention(observation_tokens)
    print("多头注意力输出形状：", tuple(multi_attention_output.shape),"多头注意力权重形状：", tuple(multi_attention_weights.shape))  # 查看输出的批次、位置数量和每个位置的特征长度。
    attention_residual_block=AttentionResidualBlock(token_dim=64,num_heads=4)
    residual_output,residual_weights=attention_residual_block(observation_tokens)
    print("残差注意力输出形状：", tuple(residual_output.shape),"残差注意力权重形状：", tuple(residual_weights.shape))  # 查看输出的批次、位置数量和每个位置的特征长度。
    position_module = LearnedPositionEmbedding(max_tokens=10, token_dim=64)  # 创建覆盖当前10个序列位置的可学习位置表。
    position = position_module(observation_tokens)  # 根据观测序列生成形状[1,10,64]的位置向量。
    transformerEncoderBlock=TransformerEncoderBlock(64,4,256)
    output_1, weight = transformerEncoderBlock(observation_tokens, position=position)  # 将内容与位置分别传入编码器。
    print("编码器输出:",tuple(output_1.shape),"权重：",tuple(weight.shape))
    batch_size, _, token_dim = output_1.shape  # 从编码器输出读取批次大小和特征长度，本例分别为2和64。
    action_count = 3  # 本次设置3个动作位置，特意与10个观测位置区分。
    action_tokens = output_1.new_zeros((batch_size, action_count, token_dim))  # 创建[B,3,64]的零内容，沿用编码器输出的设备和数据类型。
    action_position_module = LearnedPositionEmbedding(action_count, token_dim).to(output_1.device)  # 创建独立的动作位置表，并放到与观测相同的设备。
    act_position = action_position_module(action_tokens)  # 查询三个动作位置的向量，得到[1,3,64]。
    cross_attention = CrossAttention(token_dim, num_heads=4).to(output_1.device)  # 创建四头跨注意力模块，本次参数仍为随机初始化。
    cross_output, cross_weights = cross_attention(action_tokens, output_1, act_position=act_position, obs_position=position)  # 动作位置提问，从编码器输出中读取观测信息。
    print("跨注意力输出形状：", tuple(cross_output.shape))  # 预期为(2,3,64)，每个动作位置得到64维新特征。
    print("跨注意力权重形状：", tuple(cross_weights.shape))  # 预期为(2,4,3,10)，表示各头的三个动作位置分别读取十个观测位置。
    decoder_block = TransformerDecoderBlock(token_dim=64, num_heads=4, hidden_dim=256).to(output_1.device)  # 创建单层解码器，并与编码器输出放在相同设备。
    decoded_tokens, decoder_self_weights, decoder_cross_weights = decoder_block(action_tokens, output_1, act_position=act_position, obs_position=position)  # 将零动作内容、编码器观测及两种位置向量传入解码器。
    print("解码器输出形状：", tuple(decoded_tokens.shape))  # 预期为(2,3,64)，三个动作位置各得到64维特征。
    print("解码器自注意力权重：", tuple(decoder_self_weights.shape))  # 预期为(2,4,3,3)，三个动作位置互相读取。
    print("解码器跨注意力权重：", tuple(decoder_cross_weights.shape))  # 预期为(2,4,3,10)，三个动作位置读取十个观测位置。
    action_head = ActionHead(token_dim=64, action_dim=2).to(decoded_tokens.device)  # 创建共享的动作输出头，并与解码器输出放在相同设备。
    predicted_actions = action_head(decoded_tokens)  # 对每个动作位置的64维特征做相同的线性映射，得到[B,3,2]。
    print("动作预测形状：", tuple(predicted_actions.shape))  # 预期为(2,3,2)，表示两个样本各预测三个二维动作。
    print("第一个样本的动作预测：", predicted_actions[0].detach().cpu().tolist())  # 查看三个动作位置的输出值，当前权重尚未训练。
    example_predictions = torch.tensor([[[2.0, 4.0], [5.0, 8.0], [100.0, 100.0]]], requires_grad=True)  # 创建教学预测，并要求PyTorch记录计算关系，以便随后求梯度。
    example_targets = torch.tensor([[[1.0, 2.0], [4.0, 6.0], [0.0, 0.0]]])  # 创建示范目标，本例最后一个动作仅用于占位。
    example_is_pad = torch.tensor([[False, False, True]], dtype=torch.bool)  # 标记最后一个动作是补齐位置，不参与损失。
    example_loss = masked_l1_loss(example_predictions, example_targets, example_is_pad)  # 调用你写的函数，计算有效坐标的平均绝对误差。
    print("损失数值：", example_loss.item())  # 预期为1.5；item只用于打印，后续求梯度仍使用example_loss。
    print("损失形状：", tuple(example_loss.shape))  # 预期为()，表示损失是一个标量张量。
    example_loss.backward()  # 从损失反向求导，计算每个预测坐标发生微小变化时，损失会怎样变化。
    print("预测值的梯度：", example_predictions.grad)  # 查看梯度，预期四个有效坐标各为0.25，补齐位置两个坐标均为0。
    target_actions = predicted_actions.new_tensor([  # 创建与模型预测设备、数据类型一致的模拟目标，仅用于观察梯度。
    [[0.2, -0.1], [0.4, 0.3], [0.0, 0.0]],  # 第一个样本有两个有效动作，最后一个位置稍后标为补齐。
    [[-0.2, 0.1], [0.1, -0.3], [0.3, 0.2]],  # 第二个样本的三个动作都有效。
    ])  # 完成[B=2,A=3,C=2]目标张量的创建。
    action_is_pad = torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool, device=predicted_actions.device)  # 标记补齐位置，本批共有五个有效动作。
    predicted_actions.retain_grad()  # 要求保留这个中间结果的梯度，方便反向传播后直接查看。
    model_loss = masked_l1_loss(predicted_actions, target_actions, action_is_pad)  # 使用模型实际产生的预测计算损失，保留从模型参数到损失的计算关系。
    print("模型预测对应的损失：", model_loss.item())  # 打印当前随机模型对模拟目标的误差，具体数值不固定。
    model_loss.backward()  # 沿计算关系反向求导，把梯度写入参与计算的模型参数。
    print("模型预测值的梯度：", predicted_actions.grad)  # 查看各预测坐标的梯度，补齐位置应为零。
    for name, parameter in [("动作头", action_head.act_proj.weight), ("解码器跨注意力V投影", decoder_block.cross_attention.v_proj.weight), ("图像投影", projector.proj.weight)]:  # 选择实际参与此次预测的三个参数。
        gradient = parameter.grad  # 读取该参数在本次反向传播中得到的梯度。
        print(name, "梯度范数：", None if gradient is None else gradient.norm().item())  # 用一个数表示整组梯度的大小；未获得梯度时显示None。

    fake_images = torch.rand(2, 3, 96, 96, device=features.device)  # 模拟两张96×96的RGB图像，取值0～1与数据集一致。
    mini_model = MiniACT(  # 创建一个参数统一注册的教学模型实例。
    image_channels=3,  # 原始图像是三通道彩色图。
    backbone_channels=8,  # 视觉骨干输出8通道特征图。
    state_dim=state.shape[1],  # 从状态张量读取状态维度。
    action_dim=2,  # PushT每个动作包含两个目标坐标。
    token_dim=64,  # 设置内部token特征长度。
    num_head=4,  # 设置注意力头数。
    num_hidden=256,  # 设置前馈网络的中间特征长度。
    action_count=3,  # 本次预测三个动作。
    max_obs_token=1 + 6 * 6,  # 一个状态token加6×6特征图上的全部位置。
    encoder_layer=2,  # 堆叠两层编码器。
    decoder_layer=1,  # 先使用一层解码器。
    latent_dim=8,  # 潜变量维度，教学用小值。
    vae_layers=2,  # VAE编码器的层数。
    ).to(features.device)  # 将模型及其注册的子模块放在输入设备。
    mini_targets = torch.rand(2, 3, 2, device=fake_images.device)  # 模拟示范动作；训练时VAE编码器需要它才能推断潜变量分布。
    mini_actions, mini_mu, mini_logvar = mini_model(fake_images, state, target_actions=mini_targets)  # 训练模式下走VAE采样分支。
    print("MiniACT输出形状：", tuple(mini_actions.shape))  # 当前配置预期为(2,3,2)。
    print("潜变量参数形状：mu", tuple(mini_mu.shape), "logvar", tuple(mini_logvar.shape))  # 预期都是(2,8)。
    mini_model.eval()  # 切换到推理模式，验证z全零的分支。
    infer_actions, infer_mu, infer_logvar = mini_model(fake_images, state)  # 推理时不提供示范动作。
    print("推理输出形状：", tuple(infer_actions.shape), "mu是否为None：", infer_mu is None)  # 预期(2,3,2)与True。
    vae_encoder = VaeEncoder(state_dim=2, action_dim=2, token_dim=64, num_head=4,
                         num_hidden=256, num_layers=2, action_count=3, latent_dim=8)
    mu, logvar = vae_encoder(state, torch.rand(2, 3, 2))
    print("mu形状：", tuple(mu.shape), "logvar形状：", tuple(logvar.shape))   # 期望都是 (2, 8)
