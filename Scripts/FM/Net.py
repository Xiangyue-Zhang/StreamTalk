import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import numpy as np
from einops import rearrange, repeat
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter



class EMA:
    """
    指数移动平均（Exponential Moving Average）
    用于稳定训练并显著提升生成质量
    """
    def __init__(self, model, decay=0.9999, update_after_step=0):
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.step = 0
        
        # 初始化 shadow parameters（仅包含可训练参数）
        self.shadow_params = [
            p.clone().detach() for p in model.parameters() if p.requires_grad
        ]
        # 保存当前参数用于恢复
        self.backup_params = []

    def update(self):
        """每步更新 EMA"""
        if self.step < self.update_after_step:
            self.step += 1
            return
        
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))  # 训练初期使用较低 decay
        
        for s_param, param in zip(self.shadow_params, [p for p in self.model.parameters() if p.requires_grad]):
            s_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def store(self):
        """备份当前模型参数"""
        self.backup_params = [p.clone() for p in self.model.parameters() if p.requires_grad]

    def restore(self):
        """恢复原始模型参数"""
        for param, b_param in zip([p for p in self.model.parameters() if p.requires_grad], self.backup_params):
            param.data.copy_(b_param.data)
        self.backup_params = []

    def copy_to(self):
        """将 EMA 参数复制到模型（用于推理/采样）"""
        for param, s_param in zip([p for p in self.model.parameters() if p.requires_grad], self.shadow_params):
            param.data.copy_(s_param.data)

    def state_dict(self):
        return {
            'shadow_params': [p.clone().detach() for p in self.shadow_params],
            'step': self.step
        }

    def load_state_dict(self, state_dict):
        self.shadow_params = [p.clone().detach() for p in state_dict['shadow_params']]
        self.step = state_dict['step']



class StyleEmbedding(nn.Module):
    def __init__(self, num_persons, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_persons, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        return self.mlp(t)

class SimpleSpeechModel(nn.Module):
    def __init__(self, 
        feature_dim=512, 
        output_feat_dim=100,
        num_person=31):
        super(SimpleSpeechModel, self).__init__()
        self.num_person = num_person
        self.pos_enc = Summer(PositionalEncoding1D(feature_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, 
                                                   nhead=4,
                                                   dim_feedforward=2 * feature_dim, 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.audio_feature_map = nn.Linear(1024 + 64, feature_dim)
        self.vertice_map_r = nn.Linear(feature_dim, output_feat_dim)
        self.output_feat_dim = output_feat_dim

        self.style_embed = StyleEmbedding(num_persons=31, dim=64)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, huberts, pid):
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(huberts.device)
        style_embed = self.style_embed(one_hots).unsqueeze(1).expand(-1, huberts.shape[1], -1)
        
        vertice_input = self.audio_feature_map(torch.cat([huberts, style_embed], dim=-1))
        vertice_input = self.pos_enc(vertice_input)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        return vertice_out




class DiTBlock2PostNorm(nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_poses_prompt = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_poses_prompt  = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)



    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        poses_prompt = kwargs['poses_prompt']
        prompt_mask = kwargs['prompt_mask']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(
            time_emb).chunk(9, dim=-1)
        # poses prompt
        poses_prompt0 = poses_prompt
        poses_prompt = self.ln_poses_prompt(poses_prompt)
        poses_prompt = poses_prompt * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        poses_prompt = self.ff_poses_prompt(poses_prompt) * gate3.unsqueeze(1) + poses_prompt0
        x = self.cross_attn(x, poses_prompt, poses_prompt, key_padding_mask=prompt_mask)[0] + x




        # time_style_embed
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)

        # ff
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x



class TimeEncoding(nn.Module):
    def __init__(self, dim, num_freqs=64):
        """
        时间编码模块，用于将连续时间 t ∈ [0,1] 映射为高维嵌入。

        参数：
        - dim: 输出嵌入的维度
        - num_freqs: 使用的频率数量（每个频率对应一个 sin 和 cos）
        """
        super(TimeEncoding, self).__init__()
        self.dim = dim
        self.num_freqs = num_freqs

        # 频率项，使用可学习的 buffer
        inv_freq = 1.0 / (10000 ** (torch.arange(0, num_freqs, dtype=torch.float32) / num_freqs))
        self.register_buffer('inv_freq', inv_freq)

        # MLP 将傅里叶特征映射到目标维度
        self.mlp = nn.Sequential(
            nn.Linear(num_freqs * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        """
        输入：
        - t: (batch_size,)，表示时间（通常归一化在 [0,1] 范围内）

        输出：
        - time_embedding: (batch_size, dim)
        """
        # 构造傅里叶特征
        x = torch.outer(t, self.inv_freq)  # (batch_size, num_freqs)
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        x = torch.cat([sin_x, cos_x], dim=-1)  # (batch_size, 2 * num_freqs)

        # 通过 MLP 映射到目标维度
        time_embedding = self.mlp(x)
        return time_embedding
    
    


    
 


class MultiModalFusion2(nn.Module):
    def __init__(self, feature_size=384, hidden_size=512):
        super(MultiModalFusion2, self).__init__()
        self.in_prj = nn.Linear(feature_size, hidden_size)
        self.out_prj = nn.Linear(hidden_size, feature_size)
        self.atten = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        # self.scale = nn.Parameter(torch.ones(1)* 1e-6) 


    def forward(self, x1, x2, x3):
        B, T = x1.shape[:2]
        x0 = torch.cat([x1, x2, x3], dim=1)

        x = self.in_prj(x0)
        x = self.atten(x, x, x)[0] 
        x = self.out_prj(x) + x0
        return x[:, :T], x[:, T:2*T], x[:, 2*T:]



class DiffusionDITNetPartsFixedExpressions2PostNormInteraction2(nn.Module):
    _CFG_CONDITION_KEYS = frozenset(("hubert", "expressions"))
    _CFG_NON_BATCH_KEYS = frozenset((
        "hand_indices",
        "upper_indices",
        "lower_indices",
        "face_joint_indices",
        "trans_indices",
    ))

    def __init__(self, hidden_size=384,  num_layers=6, num_person=31, expression_dim=100):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 , hidden_size)
        self.in_trans = nn.Linear(3, hidden_size)

        self.in_x_hand = nn.Linear(55 * 6 + 3 + expression_dim, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 3 + expression_dim, hidden_size)
        self.in_x_trans = nn.Linear(55 * 6 + 3 + expression_dim, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.trans_fuse = nn.Linear(hidden_size * 2, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_trans = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])

        self.fusion_blocks = nn.ModuleList([MultiModalFusion2(feature_size=hidden_size) for _ in range(num_layers-1)])

        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6)
        self.trans = nn.Linear(hidden_size, 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6, hidden_size)
        self.seed_prj_trans = nn.Linear(3, hidden_size)



        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_joint_indices = kwargs['face_joint_indices']
        trans_indices = kwargs['trans_indices']
        pid = kwargs['pid']
        poses_prompt = kwargs['poses_prompt']
        prompt_mask = kwargs['prompt_mask']
        trans_prompt = kwargs['trans_prompt']
        trans_mask = kwargs['trans_mask']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]
        trans = x[...,trans_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        trans = self.in_trans(trans)
        
        x = torch.cat([x, expressions], dim=-1)
        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)
        xtrans = self.in_x_trans(x)



        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))
        trans = self.trans_fuse(torch.cat([trans, xtrans], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        hand_poses_prompt = self.seed_prj_hand(poses_prompt[..., hand_indices])
        body_poses_prompt = self.seed_prj_body(poses_prompt[..., body_indices])
        trans_prompt = self.seed_prj_trans(trans_prompt)


        hand_poses_prompt = self.pos_enc(hand_poses_prompt)
        body_poses_prompt = self.pos_enc(body_poses_prompt)
        trans_prompt = self.pos_enc(trans_prompt)


        # hand = self.pos_enc(hand)
        # body = self.pos_enc(body)
        # trans = self.pos_enc(trans)

        all = torch.cat([hand, body,trans], dim=1)
        all = self.pos_enc(all)
        hand, body, trans = all.chunk(3, dim=1)

        nid = 0
        for hand_blk, body_blk, trans_blk in zip(self.transformer_hand, self.transformer_body, self.transformer_trans):
            kwargs = {'poses_prompt': hand_poses_prompt, 'time_embed': time_style_embed, 'prompt_mask': prompt_mask}
            hand = hand_blk(hand, **kwargs)

            kwargs = {'poses_prompt': body_poses_prompt, 'time_embed': time_style_embed, 'prompt_mask': prompt_mask}
            body = body_blk(body, **kwargs)

            kwargs = {'poses_prompt': trans_prompt, 'time_embed': time_style_embed, 'prompt_mask': trans_mask}
            trans = trans_blk(trans, **kwargs)

            if nid < len(self.fusion_blocks):
                fuse_block = self.fusion_blocks[nid]
                hand, body, trans = fuse_block(hand, body, trans)
            nid = nid + 1





        hand = self.hand(hand)
        body = self.body(body)
        trans = self.trans(trans)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        x[..., trans_indices] = trans
        return x

    @staticmethod
    def _cfg_scale_as_float(cfg_scale):
        if torch.is_tensor(cfg_scale):
            if cfg_scale.numel() != 1:
                raise ValueError("cfg_scale must be a scalar")
            return float(cfg_scale.detach().item())
        return float(cfg_scale)

    def _cfg_kwargs(self, kwargs, batch_size, pair_batch, expressions_uncond):
        cfg_kwargs = {}
        for key, value in kwargs.items():
            if key in self._CFG_CONDITION_KEYS:
                if not torch.is_tensor(value):
                    raise TypeError(f"{key} must be a torch.Tensor for CFG")
                unconditional = (
                    torch.zeros_like(value)
                    if key == "hubert"
                    else expressions_uncond
                )
                cfg_kwargs[key] = (
                    torch.cat((unconditional, value), dim=0)
                    if pair_batch
                    else unconditional
                )
                continue

            is_batch_tensor = (
                key not in self._CFG_NON_BATCH_KEYS
                and torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == batch_size
            )
            cfg_kwargs[key] = (
                torch.cat((value, value), dim=0)
                if pair_batch and is_batch_tensor
                else value
            )
        return cfg_kwargs

    def forward_with_cfg(self, x, time_steps, cfg_scale=1.0, **kwargs):
        """Run classifier-free guidance without changing the cfg=1 path."""
        cfg_scale = self._cfg_scale_as_float(cfg_scale)
        conditioned_kwargs = dict(kwargs)
        expressions_uncond = conditioned_kwargs.pop("expressions_uncond", None)

        if cfg_scale == 1.0:
            return self.forward(x, time_steps, **conditioned_kwargs)

        expressions = conditioned_kwargs.get("expressions")
        if not torch.is_tensor(expressions_uncond):
            raise ValueError(
                "expressions_uncond is required as a torch.Tensor when "
                "cfg_scale != 1"
            )
        if not torch.is_tensor(expressions):
            raise TypeError("expressions must be a torch.Tensor for CFG")
        if (
            expressions_uncond.shape != expressions.shape
            or expressions_uncond.dtype != expressions.dtype
            or expressions_uncond.device != expressions.device
        ):
            raise ValueError(
                "expressions_uncond must match expressions shape, dtype, "
                "and device"
            )

        batch_size = x.shape[0]
        if cfg_scale == 0.0:
            unconditional_kwargs = self._cfg_kwargs(
                conditioned_kwargs,
                batch_size=batch_size,
                pair_batch=False,
                expressions_uncond=expressions_uncond,
            )
            return self.forward(x, time_steps, **unconditional_kwargs)

        if not torch.is_tensor(time_steps) or time_steps.ndim == 0:
            raise ValueError("time_steps must be a batched torch.Tensor for CFG")
        if time_steps.shape[0] != batch_size:
            raise ValueError(
                "time_steps and x must have the same leading batch dimension"
            )

        paired_x = torch.cat((x, x), dim=0)
        paired_time_steps = torch.cat((time_steps, time_steps), dim=0)
        paired_kwargs = self._cfg_kwargs(
            conditioned_kwargs,
            batch_size=batch_size,
            pair_batch=True,
            expressions_uncond=expressions_uncond,
        )
        paired_prediction = self.forward(
            paired_x, paired_time_steps, **paired_kwargs
        )
        unconditional, conditional = paired_prediction.chunk(2, dim=0)
        return unconditional + cfg_scale * (conditional - unconditional)
    
    def sample(self, noise, normalize_func=None, cfg_scale=1.0, **kwargs):
        res_list = []
        B, L = noise.shape[:2]
        device = noise.device
        n_timesteps = 10
        time_steps = torch.linspace(0, 1, steps=n_timesteps + 1, device=device)
        x0 = noise
        xt = x0
        res_list.append(xt)
        for i in range(n_timesteps):
            ts, te = time_steps[i], time_steps[i + 1]
            ts = ts.expand(B)
            te = te.expand(B)
            pred_x1 = self.forward_with_cfg(
                xt, ts, cfg_scale=cfg_scale, **kwargs
            )
            if normalize_func is not None:
                pred_x1 = torch.cat([normalize_func(pred_x1[...,:330]), pred_x1[...,330:]], dim=-1)
            xt = x0 * (1 - te[:,None, None]) + pred_x1 * te[:,None, None]
            res_list.append(xt)
        return res_list
    
    def sample_method2(self, noise, normalize_func=None, cfg_scale=1.0, **kwargs):
        res_list = []
        B, L = noise.shape[:2]
        device = noise.device
        n_timesteps = 10
        time_steps = torch.linspace(0, 1, steps=n_timesteps + 1, device=device)
        x0 = noise
        xt = x0
        res_list.append(xt)
        for i in range(n_timesteps):
            ts, te = time_steps[i], time_steps[i + 1]
            ts = ts.expand(B)
            te = te.expand(B)
            pred_x1 = self.forward_with_cfg(
                xt, ts, cfg_scale=cfg_scale, **kwargs
            )
            if normalize_func is not None:
                pred_x1 = torch.cat([normalize_func(pred_x1[...,:330]), pred_x1[...,330:]], dim=-1)
            # xt = x0 * (1 - te[:,None, None]) + pred_x1 * te[:,None, None]
            xt = xt + (pred_x1 - xt)  / (1.0 - ts[:,None, None]) * (te - ts)[:,None, None]
            res_list.append(xt)
        return res_list

    def sample_bk(self, noise, normalize_func=None, cfg_scale=1.0, **kwargs):
        B, L = noise.shape[:2]
        device = noise.device
        n_timesteps = 10
        time_steps = torch.linspace(0, 1, steps=n_timesteps + 1, device=device)
        x0 = noise
        xt = x0
        for i in range(n_timesteps):
            ts, te = time_steps[i], time_steps[i + 1]
            ts = ts.expand(B)
            te = te.expand(B)
            pred_x1 = self.forward_with_cfg(
                xt, ts, cfg_scale=cfg_scale, **kwargs
            )
            if normalize_func is not None:
                pred_x1 = torch.cat([normalize_func(pred_x1[...,:330]), pred_x1[...,330:]], dim=-1)
            xt = x0 * (1 - te[:,None, None]) + pred_x1 * te[:,None, None]

        return xt

    def sample_with_kp(
        self, noise, original_x1, normalize_func=None, cfg_scale=1.0, **kwargs
    ):
        B, L = original_x1.shape[:2]
        device = original_x1.device
        original_n_timesteps = 10
        original_time_steps = torch.linspace(0, 1, steps=original_n_timesteps + 1, device=device)
        kp_n_timesteps = 5
        # kp_time_steps = torch.linspace(original_time_steps[-(kp_n_timesteps + 1)], 1, steps=kp_n_timesteps + 1, device=device)
        kp_time_steps = torch.linspace(original_time_steps[-2], 1, steps=kp_n_timesteps + 1, device=device)
        x0 = noise
        xt = x0 * (1 - kp_time_steps[1]) + original_x1 * kp_time_steps[1]
        for i in range(1, kp_n_timesteps):
            ts, te = kp_time_steps[i], kp_time_steps[i + 1]
            ts = ts.expand(B)
            te = te.expand(B)
            pred_x1 = self.forward_with_cfg(
                xt, ts, cfg_scale=cfg_scale, **kwargs
            )
            if normalize_func is not None:
                pred_x1 = torch.cat([normalize_func(pred_x1[...,:330]), pred_x1[...,330:]], dim=-1)
            xt = x0 * (1 - te[:,None, None]) + pred_x1 * te[:,None, None]

        return xt

    def sample_with_kp_bk(
        self,
        noise,
        origin_xt,
        idx,
        normalize_func=None,
        cfg_scale=1.0,
        **kwargs,
    ):
        B, L = origin_xt.shape[:2]
        device = origin_xt.device
        original_n_timesteps = 10
        original_time_steps = torch.linspace(0, 1, steps=original_n_timesteps + 1, device=device)
        x0 = noise
        xt = origin_xt
        for i in range(idx, original_n_timesteps):
            ts, te = original_time_steps[i], original_time_steps[i + 1]
            ts = ts.expand(B)
            te = te.expand(B)
            pred_x1 = self.forward_with_cfg(
                xt, ts, cfg_scale=cfg_scale, **kwargs
            )
            if normalize_func is not None:
                pred_x1 = torch.cat([normalize_func(pred_x1[...,:330]), pred_x1[...,330:]], dim=-1)
            xt = x0 * (1 - te[:,None, None]) + pred_x1 * te[:,None, None]

        return xt
    



 






def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    pass
    
