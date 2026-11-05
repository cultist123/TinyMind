import math
from typing import Optional,Tuple,Union
import torch
import torch.nn as nn
from model import TinyMindConfig
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    #init初始化
    def __init__(self,dim:int,eps:float=1E-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    # norm
    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim:int,end:int=32*1024,rope_base:float=1e6,rope_scaling:Optional[dict]=None):
    #初始化rope频率
    freqs,attn_factor=(1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))),1.0

    #需要用到yarn
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow=(
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

        #推断的长度大于训练长度, 使用缩放
        if end > orig_max:
            #波长b到i的映射
            inv_dim=lambda b :(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
            #划分高低维度
            #fast:不需要缩放的高频部分
            #slow:需要缩放的低频部分
            fast=max(math.floor(inv_dim(beta_fast)),0)
            slow=min(math.ceil(inv_dim(beta_slow)),dim//2 -1)

            #计算缩放因子
            #fast之前,ramp为0,在slow之后,ramp为1,在(fast,slow)之间,线性过渡
            ramp = torch.clamp(
                (torch.arange(dim//2,device=freqs.device).float()-fast)
                /max(slow-fast,0.001),
                0,
                1,
            )

            # 当 ramp=0 时（高频/快组）：系数为 1，保持原频率不变
            # 当 ramp=1 时（低频/慢组）：系数为 1/factor，频率缩小 factor 倍（波长拉长），即位置插值
            # 当 ramp 在 0~1 之间时：在两者之间线性过渡
            freqs=freqs*(1-ramp+ramp/factor)

    #根据end,生成位置索引t
    t=torch.arange(end,device=freqs.device).float()
    #计算外积,将t和频率部分相乘,得到每个位置的旋转角度
    freqs=torch.outer(t,freqs).float()
    freqs_cos=torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)*attn_factor
    freqs_sin=torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)*attn_factor

    return freqs_cos,freqs_sin

#编写RoPE
def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    #[a,b]->[-b,a]
    def rotate_half(x):
        #x.shape[-1]取最后一个维度的大小
        #x[...,x.shape[-1]//2:]取出x的后半部分
        #x[...,:x.shape[-1]//2]取出x的前半部分
        return torch.cat(
            [-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]],dim=-1
            )
    #x_rotated=x*cos+rotate_half(x)*sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed,k_embed
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,:,:,None,:]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, n_rep * num_key_value_heads, head_dim)
    )

#GQA实现
class Attention(nn.Module):
    def __init__(self, args:TinyMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0 
        # 计算每个key-value头对应的query头数量,必须是整数倍,不然报错

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads//self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        #投影层
        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.o_proj=nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias=False)

        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention
    def forward(
    self,
    x: torch.Tensor,
    position_embdding: Tuple[torch.Tensor, torch.Tensor],
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache=False,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    #投影:linear计算q,k,v
        bsz, seq_len, _ = x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
    #把输入拆分成多个头,用view
    #现在是512维,有8个头,所以每个头拆成64维
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    #q和k,使用RoPE
        cos, sin = position_embdding
        xq,xk=apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
    #对于k和v,使用repeat(注意kv cashe)
        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_key_value = (xk,xv)if use_cache else None

        xq,xk,xv=(
            xq.transpose(1,2),
            # [bsz, n_local_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1,2),
            repeat_kv(xv, self.n_rep).transpose(1,2),
        )
    #进行attention计算,q@k^T/sqrt(d)
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask==1)):
            attn_mask=(
                None
                if attention_mask is None 
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            scores = (xq@xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            scores = scores+torch.triu(
                torch.full((seq_len,seq_len), float('-inf'), device=scores.device), 
                diagonal=1
                ).unsqueeze(0).unsqueeze(0)
    #最后拼接头,输出投影,返回
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            scores = F.softmax(scores.float(),dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
    #[bsz, seq_len, n_local_heads*head_dim]
        output=output.transpose(1,2).reshape(bsz, seq_len, -1)
        output=self.resid_dropout(self.o_proj(output))
        return output, past_key_value

#FFN实现
class FeedForward(nn.Module):
    #初始化
    def __init__(self,args:TinyMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8/3)
            args.intermediate_size = 64*((intermediate_size + 64-1) // 64)
    #升维
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
    #降维
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
    #门控
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
    #dropout
        self.dropout = nn.Dropout(args.dropout)
    #激活函数
        self.act_fn = ACT2FN[args.hidden_act]
    def forward(self, x):  
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))

#block实现
class TinyMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:TinyMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id=layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attetion_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self,hidden_states,position_embeddings,post_key_value=None,use_cache=False,attention_mask=None):
        residual = hidden_states
        hidden_states,present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            post_key_value,
            use_cache,
            attention_mask
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attetion_layernorm(hidden_states)
            )
        return hidden_states, present_key_value

#model 实现
class TinyMindModel(nn.Module):
    def __init__(self, config: TinyMindConfig):
        super().__init__()
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        self.layers=nn.ModuleList(
            [TinyMindBlock(i, config) for i in range(config.num_hidden_layers)]
            )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #RoPE预计算
        freqs_cos,freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
            )
        
        self.register_buffer("freqs_cos", freqs_cos,persistent=False)
        self.register_buffer("freqs_sin", freqs_sin,persistent=False)

    def forward(
            self, 
            input_ids:[Optional[torch.Tensor]]=None,
            attention_mask:[Optional[torch.Tensor]]=None,
            past_key_values:[Optional[Tuple[Tuple[torch.Tensor]]]]=None,
            use_cache:bool=False,
            **kwargs,
        ):
            batch_size, seq_length = input_ids.shape
            if hasattr(past_key_values, "layers"):
                past_key_values = None

            past_key_values = past_key_values or [None] * len(self.layers)
            start_pos=(
                past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
            )
            hidden_states = self.dropout(self.embed_tokens(input_ids))
            position_embeddings = (
                self.freqs_cos[start_pos:start_pos+seq_length], 
                self.freqs_sin[start_pos:start_pos+seq_length]
                )
    
            presents = []
            for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
                hidden_states, present = layer(
                    hidden_states,
                    position_embeddings,
                    post_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
                presents.append(present)

            hidden_states = self.norm(hidden_states)
            return hidden_states, presents

#模型进行完整的封装
class TinyMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyMindConfig

    def __init__(self,config: TinyMindConfig):
        self.config = config

        super().__init__(config)

        self.model = TinyMindModel(config)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False
        )

        #权重共享
        #输出层的权重和嵌入层的权重共享
        self.model.embed_tokens.weight = self.lm_head.weight

        self.OUT=CausalLMOutputWithPast()#封装模型的输出

    def forward(self,input_ids:[Optional[torch.Tensor]]=None,
                attention_mask:[Optional[torch.Tensor]]=None,
                past_key_values:[Optional[Tuple[Tuple[torch.Tensor]]]]=None,
                use_cache:bool=False,
                logits_to_keep:Union[int,torch.Tensor]=0,
                labels:Optional[torch.Tensor]=None,
                **args,
            ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args, 
        )
        #logits to keep是整数,那就保留最后n个位置
        #生成的时候只要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_to_keep,None) 
            if isinstance(logits_to_keep,int) 
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__("hidden_states", hidden_states)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", past_key_values)
        #有labels时计算loss:用第t个位置的logits预测第t+1个token
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        self.OUT.__setitem__("loss", loss)
        #还没有实现MoE,aux_loss为0
        self.OUT.__setitem__("aux_loss", torch.zeros((), device=logits.device))
        return self.OUT

