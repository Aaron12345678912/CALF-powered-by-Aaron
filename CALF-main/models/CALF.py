import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from models.GPT2_arch import AccustumGPT2Model

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12):
        super(Encoder_PCA, self).__init__()
        # Linear projection → 产生 Q
        self.linear = nn.Linear(input_dim, hidden_dim)

        # 文本分支交叉注意力: Q=原始实测信号, K=V=词嵌入 (冻结保持语义)
        self.word_cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # word_embedding 可能是 (d_model, N) 或 (N, d_model)，统一转为 (N, d_model)
        if word_embedding.shape[0] == hidden_dim:
            self.word_embedding = word_embedding.T  # (d_model, N) → (N, d_model)
        else:
            self.word_embedding = word_embedding  # 已经是 (N, d_model)

    def forward(self, x, x_TQ=None):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            word_emb = self.word_embedding.repeat(B, 1, 1)  # (B, N, d_model)
        elif self.word_embedding.shape[0] != B:
            word_emb = self.word_embedding[0].repeat(B, 1, 1)
        else:
            word_emb = self.word_embedding

        # Linear 投影 → Q
        x_proj = self.linear(x)  # (B, M, d_model)

        # 与 TQ 特征进行残差融合并自动退化
        if x_TQ is not None:
            x_fused_tq = x_proj + x_TQ
        else:
            x_fused_tq = x_proj

        # 文本分支交叉注意力: Q 替换为残差融合特征, K,V 来自词嵌入（冻结）
        q = x_fused_tq.transpose(0, 1)
        k = v = word_emb.transpose(0, 1)
        x_text, _ = self.word_cross_attention(q, k, v)
        x_text = x_text.transpose(0, 1)

        return x_fused_tq, x_text

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
    
        self.task_name = configs.task_name
        self.gpt2 = AccustumGPT2Model.from_pretrained('./models/gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_text = AccustumGPT2Model.from_pretrained('./models/gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]#通过gpt_layers裁剪层数
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)#LORA微调参数传入
        
        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])

        # ---- TQ 模块 ----
        self.seq_len = configs.seq_len
        self.cycle_len = configs.cycle
        self.enc_in = configs.enc_in

        # 1. 基础初始化：依然使用微小随机白噪声
        init_tq = torch.randn(self.cycle_len, self.enc_in) * 0.02
        
        # 2. 物理注入 (Physics-Informed Prior)
        # 仅仅作为优良的“起跑线”，训练过程中依然 100% 跟随梯度微调！
        if self.cycle_len == 144:
            with torch.no_grad():
                t = torch.arange(self.cycle_len).float()
                # 构建一个开口向下、顶点在中午(72)的抛物线形状先验，夜晚（0-35和110-143）被 clamp 截断为极小值
                parabola = torch.clamp(1.0 - ((t - 72.0) / 36.0)**2, min=0.0).unsqueeze(1).repeat(1, self.enc_in)
                # 叠加先验，让它一上来就拥有昼夜规律的轮廓
                init_tq = init_tq + parabola * 0.1

        # 3. 封装为 Parameter，并开启 requires_grad=True 允许跟随梯度更新
        self.temporalQuery = torch.nn.Parameter(init_tq, requires_grad=True)
        
        # 物理流投影层：将原始 seq_len 独立投影到大模型维度 d_model
        self.tq_proj = nn.Linear(self.seq_len, configs.d_model)
        # -----------------

        self.in_layer = Encoder_PCA(
            configs.seq_len,                # 输入维度（线性层输入）
            word_embedding,                 # 用于交叉注意力的词嵌入
            hidden_dim=configs.d_model,     # d_model
        )
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj, self.tq_proj):
            layer.to(device=device)
            layer.train()
        


    def forecast(self, x, cycle_index=None):
        B, L, M = x.shape

        # [RevIN 标准化注释已标明]
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        # x /= stdev


        # ---- TQ 模块：在标准化后的数据上提取周期模式 ----
        if cycle_index is not None:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=x.device).view(1, -1)) % self.cycle_len  # (B, seq_len)
        else:
            gather_index = torch.arange(self.seq_len, device=x.device) \
                .unsqueeze(0).expand(B, -1) % self.cycle_len  # (B, seq_len)

        # temporalQuery: (cycle_len, enc_in) -> 收集得到 (B, seq_len, enc_in)
        query_input = self.temporalQuery[gather_index]  # (B, seq_len, enc_in)
        
        # 转换维度为 (Batch, Channel, Seq_Len) 
        x = rearrange(x, 'b l m -> b m l')  # (B, enc_in, seq_len)
        query_tq_input = query_input.permute(0, 2, 1) # (B, enc_in, seq_len)
        
        # 物理流：得到绝对纯净、不受脏数据污染的刚性基线特征 x_TQ
        x_TQ = self.tq_proj(query_tq_input) # (B, enc_in, d_model)
        
        # ---- 3. 调用重构后的 PCA 层进行残差融合 ----
        # x_fusion: 融合特征 x_fused_tq (x_proj + x_TQ)
        # x_text: 交叉注意力结果（Query 为 x_fused_tq）
        x_fusion, x_text = self.in_layer(x, x_TQ=x_TQ) # 均为 (B, enc_in, d_model)
        # ----------------------------------------

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=x_fusion)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=x_text)
        # residue connection
        outputs_time = outputs_time + x_fusion
        outputs_text = outputs_text + x_text
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        # [RevIN 反标准化注释已标明]
        # outputs_text = outputs_text * stdev + means
        # outputs_time = outputs_time * stdev + means


        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }


    def classification(self, x):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)
        
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        
        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)
        
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)
        
        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }
    

    def imputation(self, x, mask):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x**2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }


    def forward(self, x, mask=None, cycle_index=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x, cycle_index=cycle_index)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
