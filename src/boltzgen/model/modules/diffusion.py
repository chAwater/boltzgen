# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
#
# 本文件实现 BoltzGen 的扩散模型核心逻辑，包括：
#   - DiffusionModule: 去噪神经网络（原子→token→Transformer→原子的 coarse-to-fine 架构）
#   - OutTokenFeatUpdate: 多步推理时的 token 特征累积器
#   - AtomDiffusion: 扩散过程的总调度器（训练加噪、采样去噪、损失计算）
#
# 整体基于 EDM（Elucidating Diffusion Models, Karras et al. 2022）框架，
# 并融合了 AlphaFold3 的蛋白质结构预条件化策略。

from __future__ import annotations

from math import sqrt
from math import exp
from scipy.stats import norm  # 用于 pred_threshold 的正态分布分位数计算
import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module
from typing import Any, Dict, Optional, List

from tqdm import tqdm
import boltzgen.model.layers.initialize as init  # 参数初始化工具（final_init_ 等）
from boltzgen.data import const  # 数据常量（链类型 ID、残基类型权重等）
from boltzgen.model.layers.miniformer import MiniformerModule
from boltzgen.model.layers.pairformer import PairformerModule
from boltzgen.model.loss.diffusion import (
    compute_bond_loss,         # 共价键长度损失
    smooth_lddt_loss,          # 可微分的局部距离差异测试（lDDT）损失
    weighted_rigid_align,      # 加权刚体对齐（Kabsch 算法），用于 MSE 计算前对齐预测与真实结构
    weighted_rigid_centering,  # 加权质心对齐（仅平移，不旋转）
)
from boltzgen.model.modules.encoders import (
    AtomAttentionDecoder,      # 原子级注意力解码器：token 表示 → 原子坐标更新
    AtomAttentionEncoder,      # 原子级注意力编码器：原子坐标 → 聚合到 token 表示
    CoordinateConditioning,
    FourierEmbedding,          # 随机傅里叶特征嵌入：将连续标量 σ 映射到高维向量
    SingleConditioning,        # 单序列条件化：融合时间编码、trunk 特征和输入特征
)
from boltzgen.model.modules.transformers import (
    ConditionedTransitionBlock,  # 带条件的前馈网络（用于 OutTokenFeatUpdate）
    DiffusionTransformer,        # 扩散 Transformer：带 AdaLN 和 pair bias 的多层自注意力
)
from boltzgen.model.modules.utils import (
    LinearNoBias,                # 无偏置线性层
    center,                      # 将坐标中心化到质心
    center_random_augmentation,  # 中心化 + 随机 SE(3) 增强（旋转 + 平移）
    compute_random_augmentation, # 生成随机旋转矩阵和平移向量
    default,                     # 如果值为 None 则返回默认值
    log,                         # 安全对数（避免 log(0)）
)
from scipy.stats import beta  # Beta 分布 CDF，用于非均匀的 noise/step scale 调度


def optionally_tqdm(iterable, use_tqdm=True, **kwargs):
    """推理时可选的进度条包装器"""
    return tqdm(iterable, **kwargs) if use_tqdm else iterable


"""
张量维度命名约定（einops 风格）:
b  - batch size（批大小）
h  - heads（注意力头数）
n  - residue sequence length（残基/token 序列长度）
m  - atom sequence length（原子序列长度，通常 m >> n，每个残基有多个原子）
nw - windowed sequence length（窗口化序列长度，用于局部注意力）
ts - feature dimension, single（单序列特征维度，如 384）
tz - feature dimension, pairwise（成对特征维度，如 128）
as - feature dimension, atompair（原子对特征维度）
az - feature dimension, atompair input（原子对输入特征维度）
"""


# ============================================================================
# DiffusionModule: 去噪神经网络
# ============================================================================
# 对应 AlphaFold3 论文中的 Algorithm 20。
# 这是扩散模型中的核心网络 F_θ，接收含噪坐标和条件信息，输出坐标更新。
# 架构采用 coarse-to-fine 策略：
#   原子级局部注意力 → 聚合到 token 级 → 全局 Transformer → 广播回原子级
# 这样既能捕获全局结构信息，又不会因原子数过多导致注意力计算爆炸。
class DiffusionModule(Module):
    """Algorithm 20."""

    def __init__(
        self,
        token_s: int,              # token（残基）级单序列特征维度，如 384
        atom_s: int,               # 原子级特征维度
        atoms_per_window_queries: int = 32,   # 原子注意力窗口大小（query 侧）
        atoms_per_window_keys: int = 128,     # 原子注意力窗口大小（key 侧，更大以覆盖更多上下文）
        sigma_data: int = 16,      # 数据分布标准差（蛋白质坐标尺度，约 16 Å）
        dim_fourier: int = 256,    # 傅里叶时间编码的输出维度
        atom_encoder_depth: int = 3,          # 原子编码器的 Transformer 层数
        atom_encoder_heads: int = 4,
        token_layers: int = 1,
        token_transformer_depth: int = 6,     # token 级 Transformer 层数（全局推理）
        token_transformer_heads: int = 8,
        use_miniformer: bool = False,
        diffusion_pairformer_args: Dict[str, Any] = None,
        atom_decoder_depth: int = 3,          # 原子解码器的 Transformer 层数
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,  # 梯度检查点：用时间换显存
        gaussian_random_3d_encoding_dim: int = 0,
        transformer_post_ln: bool = False,
        tfmr_s: Optional[int] = None,  # Transformer 内部维度，默认 2 * token_s
        predict_res_type: bool = False,  # 是否同时预测残基类型（设计任务）
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing
        # Transformer 内部维度默认为 token 特征维度的 2 倍，以提供更大的表达能力
        if tfmr_s is None:
            tfmr_s = 2 * token_s
        self.tfmr_s = tfmr_s

        # ---- 条件化模块 ----
        # 将时间步 σ（噪声水平）编码并融合到 trunk 特征中
        # 输出 s: (B, N, tfmr_s) 作为后续 Transformer 的条件信号
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            tfmr_s=tfmr_s,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        # ---- 原子级编码器 ----
        # 在序列局部窗口内做原子级注意力，然后聚合到 token 级
        # 这是 coarse-to-fine 的第一步：原子 → token
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            gaussian_random_3d_encoding_dim=gaussian_random_3d_encoding_dim,
            transformer_post_layer_norm=transformer_post_ln,
            tfmr_s=tfmr_s,
            use_qk_norm=use_qk_norm,
        )

        # 将 SingleConditioning 的输出 s 投影后加到 token 表示 a 上
        # final_init_ 将权重初始化为接近零，使初始阶段这个残差分支几乎不贡献，保证训练稳定
        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(tfmr_s), LinearNoBias(tfmr_s, tfmr_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer_layers = nn.ModuleList()
        self.token_pairformer_layers = nn.ModuleList()

        # ---- Token 级 Transformer ----
        # 在 token（残基）级别做全局自注意力，捕获长程相互作用
        # 每层包含: AdaLN（以 s 为条件）→ pair-bias attention → conditioned FFN
        self.token_transformer = DiffusionTransformer(
            dim=tfmr_s,
            dim_single_cond=tfmr_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            use_qk_norm=use_qk_norm,
        )

        self.a_norm = nn.LayerNorm(tfmr_s)

        # ---- 原子级解码器 ----
        # 将 token 表示广播回原子级，输出坐标更新 r_update: (B, M, 3)
        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            tfmr_s=tfmr_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            predict_res_type=predict_res_type,
            use_qk_norm=use_qk_norm,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts'] — 输入序列特征（来自 embedding 层）
        s_trunk,   # Float['b n ts'] — trunk 网络输出（经 PairFormer 等处理后的全局特征）
        r_noisy,   # Float['bm m 3'] — 含噪原子坐标（bm = batch * multiplicity）
        times,     # Float['bm 1 1'] — 当前噪声水平的编码 c_noise(σ)
        feats,     # 特征字典（包含 token_pad_mask、atom_to_token 等）
        diffusion_conditioning,  # 预计算的条件信息（注意力偏置、原子编码等）
        multiplicity=1,  # 每个样本生成几个候选（multiplicity > 1 时并行去噪）
    ):
        # ---- 第一步：时间条件化 ----
        # 将噪声水平 σ 编码并融入 trunk 特征，得到时间感知的条件信号 s
        # repeat_interleave 是因为 multiplicity > 1 时，同一个输入要生成多个候选
        if self.activation_checkpointing:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # ---- 第二步：原子级编码 → 聚合到 token 级 ----
        # 在局部窗口（32 queries × 128 keys）内做原子级注意力
        # 输出 a: (B, N, tfmr_s) 是聚合后的 token 表示
        # q_skip, c_skip: 原子级跳跃连接，后续解码器会用到
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,
            multiplicity=multiplicity,
        )

        # ---- 第三步：融合条件信号并做全局 Transformer ----
        # 将时间条件 s 投影后加到 token 表示 a 上（残差连接）
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)

        # token 级全局自注意力（6 层），每层都以 s 为条件（通过 AdaLN）
        # bias 来自 pair representation，提供残基间的成对关系信息
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning["token_trans_bias"].float(),
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # ---- 第四步：token 表示广播回原子级 → 预测坐标更新 ----
        # 将 token 级信息分发到每个原子，结合跳跃连接 q_skip/c_skip
        # 输出 r_update: (B, M, 3) — 预测的坐标修正量
        r_update, res_type = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return {
            "r_update": r_update,          # 坐标更新量，会被 c_out(σ) 缩放后使用
            "token_a": a.detach(),         # token 表示（detach 因为仅用于辅助输出，不参与梯度）
            "res_type": res_type,
        }


# ============================================================================
# OutTokenFeatUpdate: Token 特征累积器
# ============================================================================
# 在多步推理中，每一步去噪都产生 token 表示（token_a）。
# 这个模块将新一步的 token 表示以时间感知的方式累积到历史中，
# 用于构建更丰富的条件信号。
class OutTokenFeatUpdate(Module):
    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        # 条件化前馈网络：以 (历史累积 + 时间编码) 为条件，处理新一步的 token 表示
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,    # 当前去噪步的噪声水平编码
        acc_a,    # 历史累积的 token 表示 (B, N, 2*token_s)
        next_a,   # 新一步的 token 表示 (B, N, 2*token_s)
    ):
        next_a = self.norm_next(next_a)
        # 将当前时间步编码为傅里叶特征，广播到所有 token 位置
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        # 拼接历史累积和时间编码作为条件
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        # 残差更新：累积表示 += f(新表示 | 条件)
        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


# ============================================================================
# AtomDiffusion: 扩散过程总调度器
# ============================================================================
# 这是扩散模型的最外层封装，负责：
#   1. 训练时：对真实坐标加噪 → 调用 DiffusionModule 去噪 → 计算损失
#   2. 推理时：从纯噪声开始 → 按照噪声调度逐步去噪 → 生成蛋白质结构
#
# 基于 EDM（Elucidating Diffusion Models）框架，核心思想是：
#   - 前向过程：x_noisy = x_clean + σ * ε（直接加高斯噪声，无需马尔可夫链）
#   - 反向过程：通过预条件化的去噪网络，在不同噪声水平下稳定预测
#   - 采样：从 σ_max 到 σ_min 的 Euler 步迭代
class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,           # DiffusionModule 的构造参数字典
        # ---- 噪声调度参数 ----
        num_sampling_steps: int = 5,   # 推理时的去噪步数（少量步数即可，因为每步计算量大）
        sigma_min: float = 0.0004,     # 最小噪声水平（接近零 → 几乎无噪声）
        sigma_max: float = 160.0,      # 最大噪声水平（远大于 σ_data → 完全掩盖数据信息）
        sigma_data: float = 16.0,      # 数据分布的标准差（蛋白质坐标尺度 ~16 Å）
        rho: float = 7,               # 调度曲线形状控制：越大 → 高噪声区域分配越多步数
        # ---- 训练噪声采样参数 ----
        # 训练时从对数正态分布 σ ~ σ_data * exp(N(P_mean, P_std²)) 采样噪声水平
        # P_mean=-1.2 意味着中位数噪声 ≈ σ_data * exp(-1.2) ≈ 4.8，偏向中等噪声
        P_mean: float = -1.2,
        P_std: float = 1.5,
        # ---- 采样器超参数 ----
        gamma_0: float = 0.8,         # 随机性控制：采样时额外注入噪声的比例
        gamma_min: float = 1.0,       # 低于此 σ 时不注入额外噪声
        noise_scale: float = 1.003,   # β（noise scale）：控制随机扰动的强度
        step_scale: float = 1.5,      # α（step scale）：控制 Euler 步的步长
        step_scale_random: list = None,  # 训练时随机选择 step_scale（增加多样性）
        # ---- 坐标增强 ----
        coordinate_augmentation: bool = True,  # 训练时做随机 SE(3) 增强（旋转+平移）
        coordinate_augmentation_inference=None,  # 推理时是否也做增强
        # ---- 对齐选项 ----
        mse_rotational_alignment: bool = False,  # MSE 损失计算前是否做刚体对齐（旋转+平移）
        alignment_reverse_diff: bool = False,    # 采样时是否对齐含噪坐标到去噪坐标
        synchronize_sigmas: bool = False,        # multiplicity > 1 时，同一样本的多个候选是否用相同 σ
        second_order_correction: bool = False,
        pass_resolved_mask_diff_train: bool = False,  # 训练时是否用 resolved mask 屏蔽未解析原子
        # ---- 调度策略 ----
        sampling_schedule: str = "af3",          # "af3" 或 "dilated"
        noise_scale_function: str = "constant",  # noise scale 随步数变化策略
        step_scale_function: str = "constant",   # step scale 随步数变化策略
        # 以下是 beta 分布调度的参数（当 function="beta" 时使用）
        min_noise_scale: float = 1.0,
        max_noise_scale: float = 1.0,
        noise_scale_alpha: float = 1.0,
        noise_scale_beta: float = 1.0,
        min_step_scale: float = 1.0,
        max_step_scale: float = 1.0,
        step_scale_alpha: float = 1.0,
        step_scale_beta: float = 1.0,
        # ---- Dilated schedule 参数 ----
        time_dilation: float = 1.0,          # 膨胀因子 λ（1.0 = 不膨胀，即退化为标准调度）
        time_dilation_start: float = 0.6,    # 关键区间起点 τ_s
        time_dilation_end: float = 0.8,      # 关键区间终点 τ_e
        pred_threshold: Optional[float] = None,  # nucleation mask 阈值
    ):
        super().__init__()
        # score_model 就是上面定义的 DiffusionModule（去噪网络 F_θ）
        self.score_model = DiffusionModule(
            **score_model_args,
        )

        # ---- 核心噪声参数 ----
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std

        # pred_threshold: 将训练噪声分布的分位数转换为 σ 阈值
        # 用于 nucleation mask——在高噪声时屏蔽某些预测，避免噪声过大时的不稳定
        if pred_threshold is None:
            self.pred_sigma_thresh = float("inf")  # 不启用
        else:
            q = norm.ppf(pred_threshold)  # 正态分布的分位数
            self.pred_sigma_thresh = self.sigma_data * exp(self.P_mean + self.P_std * q)

        self.num_sampling_steps = num_sampling_steps
        self.sampling_schedule = sampling_schedule
        self.time_dilation = time_dilation
        self.time_dilation_start = time_dilation_start
        self.time_dilation_end = time_dilation_end
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.noise_scale_function = noise_scale_function
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.noise_scale_alpha = noise_scale_alpha
        self.noise_scale_beta = noise_scale_beta
        self.step_scale = step_scale
        self.step_scale_function = step_scale_function
        self.min_step_scale = min_step_scale
        self.max_step_scale = max_step_scale
        self.step_scale_alpha = step_scale_alpha
        self.step_scale_beta = step_scale_beta
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.mse_rotational_alignment = mse_rotational_alignment
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.second_order_correction = second_order_correction
        self.pass_resolved_mask_diff_train = pass_resolved_mask_diff_train
        self.token_s = score_model_args["token_s"]

        # 注册一个零张量，用于在某些损失分支不激活时返回零值
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # ========================================================================
    # EDM 预条件化函数 (Karras et al. 2022, Table 1)
    # ========================================================================
    # 这四个函数是 EDM 框架的核心，让网络在任意噪声水平 σ 下都能稳定学习。
    #
    # 最终的去噪预测为：
    #   D(x, σ) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x, c_noise(σ))
    #
    # 其中 F_θ 是网络（DiffusionModule），x 是含噪坐标。
    #
    # 直觉理解：
    #   σ 很大时（纯噪声）: c_skip ≈ 0, c_out ≈ σ_data → 完全依赖网络预测
    #   σ 很小时（几乎无噪声）: c_skip ≈ 1, c_out ≈ 0 → 直接返回输入（已经够干净了）
    #   σ = σ_data 时: c_skip = 0.5, 输入和网络各贡献一半

    def c_skip(self, sigma):
        """跳跃连接权重：σ_data² / (σ² + σ_data²)
        σ 越小 → c_skip 越接近 1 → 越信任输入本身"""
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """网络输出缩放：σ · σ_data / √(σ² + σ_data²)
        使网络输出的方差在不同 σ 下保持一致"""
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        """输入缩放：1 / √(σ² + σ_data²)
        将含噪输入的方差归一化到 ~1（因为 Var(x_noisy) = σ² + σ_data²）"""
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        """噪声水平编码：log(σ / σ_data) × 0.25
        将 σ 映射到对数空间并缩放，作为网络的时间条件输入。
        注意：AF3 除以 σ_data 而原始 EDM 不除——这只是改变了对数的偏移量"""
        return (
            log(sigma / self.sigma_data) * 0.25
        )

    # ========================================================================
    # 预条件化网络前向传播
    # ========================================================================
    def preconditioned_network_forward(
        self,
        noised_atom_coords,  # Float['b m 3'] — 含噪原子坐标
        sigma,               # Float['b'] | float — 当前噪声水平
        network_condition_kwargs: dict,  # 条件信息（trunk 特征、pair 信息等）
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        # 如果 sigma 是标量，扩展为批次向量
        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        # 扩展维度以便与 (B, M, 3) 的坐标做广播乘法
        padded_sigma = rearrange(sigma, "b -> b 1 1")

        # 可选：训练时屏蔽未解析的原子坐标（如缺失残基的原子）
        if training and self.pass_resolved_mask_diff_train:
            res_mask = (
                network_condition_kwargs["feats"]["atom_resolved_mask"]
                .unsqueeze(-1)
                .float()
            )
            noised_atom_coords = noised_atom_coords * res_mask.repeat_interleave(
                network_condition_kwargs["multiplicity"], 0
            )

        # 调用去噪网络 F_θ：
        #   输入 = c_in(σ) × 含噪坐标（归一化方差）
        #   时间条件 = c_noise(σ)（告诉网络当前噪声水平）
        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        # 预条件化组合：
        #   去噪结果 = c_skip(σ) × 含噪输入 + c_out(σ) × 网络预测
        # 高噪声时 c_skip≈0：完全依赖网络预测
        # 低噪声时 c_skip≈1：几乎直接返回输入（网络只做微小修正）
        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )

        return denoised_coords, net_out

    # ========================================================================
    # 噪声调度（Noise Schedule）
    # ========================================================================
    # 定义从 σ_max（纯噪声）到 σ_min（近似无噪声）的递减序列。
    # 调度策略决定了每一步"降噪多少"——这对生成质量至关重要。

    def sample_schedule_af3(self, num_sampling_steps=None):
        """AlphaFold3 风格的噪声调度。
        公式: σ_i = σ_data * (σ_max^(1/ρ) + τ_i * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
        其中 τ_i = i/(N-1) 在 [0,1] 上均匀分布。
        ρ=7 使得在高噪声区域（σ 大时）步距更小，分配更多步数——
        因为高噪声时需要更精细的去噪来确定整体拓扑结构。"""
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        # τ_i: 从 0 到 1 的均匀步进
        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        # 在 σ^(1/ρ) 空间中线性插值，然后再取 ρ 次幂还原
        # 这比直接线性插值 σ 更合理，因为 σ 跨越了 4 个数量级（0.0004 ~ 160）
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        # AF3 特有：乘以 σ_data。原始 EDM 不做这一步。
        # 效果是将噪声水平校准到数据的实际尺度
        sigmas = sigmas * self.sigma_data

        # 末尾补 0，表示最后一步目标是完全无噪声
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    def sample_schedule_dilated(self, num_sampling_steps=None):
        """膨胀调度（Dilated Schedule）—— BoltzGen 的创新。
        在标准调度基础上，对关键区间 [τ_s, τ_e] 进行膨胀（增加采样密度），
        区间外相应压缩，总步数不变。
        目的：BoltzGen 用连续几何编码表示氨基酸类型，类型决定集中在
        τ ∈ [0.6, 0.8] 的窗口。膨胀调度在这个窗口分配更多步数，
        让模型在"选择氨基酸"这一关键阶段有更精细的控制。"""
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        # τ: 从 0 到 1 的均匀步进
        ts = steps / (num_sampling_steps - 1)

        # 膨胀映射函数：将 [0,1] 重新映射到 [0,1]，但关键区间被拉伸
        def dilate(ts, start, end, dilation):
            """分段线性映射:
            - [0, start) 区间被压缩（步距变大，走得快）
            - [start, end) 区间被膨胀 dilation 倍（步距变小，走得慢）
            - [end, 1] 区间被压缩
            约束: 映射前后都是 [0,1] → [0,1]，保证总步数不变"""
            x = end - start       # 原始关键区间长度
            l = start             # 关键区间左侧长度
            u = 1 - end           # 关键区间右侧长度
            assert (dilation - 1) * x <= l + u, "dilation too large"

            inv_dilation = 1 / dilation
            # ratio: 非关键区间的压缩比（< 1 表示被压缩）
            ratio = (l + u + (1 - dilation) * x) / (l + u)
            inv_ratio = 1 / ratio
            # 变换后空间中的区间边界
            lprime = l * ratio        # 左侧区间在变换后空间的长度
            uprime = u * ratio
            xprime = x * dilation     # 关键区间膨胀后的长度

            # 三段分段线性函数（无分支，用布尔掩码实现向量化计算）
            lower_third = ts * inv_ratio                              # 左段：加速通过
            middle_third = (ts - lprime) * inv_dilation + l           # 中段：减速（密集采样）
            upper_third = (ts - (lprime + xprime)) * inv_ratio + l + x  # 右段：加速通过
            return (
                (ts < lprime) * lower_third
                + ((ts >= lprime) & (ts < lprime + xprime)) * middle_third
                + (ts >= lprime + xprime) * upper_third
            )

        # 应用膨胀映射：将均匀 τ 重新分配
        dilated_ts = dilate(
            ts, self.time_dilation_start, self.time_dilation_end, self.time_dilation
        )
        # 用映射后的 τ 计算 σ（与 AF3 调度相同的公式，只是 τ 分布不同）
        sigmas = (
            self.sigma_max**inv_rho
            + dilated_ts * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    # ========================================================================
    # 可变 scale 调度（基于 Beta 分布 CDF）
    # ========================================================================
    # noise_scale（β）和 step_scale（α）可以随去噪步数变化，
    # 而不是保持常数。Beta CDF 提供了灵活的 S 形曲线。

    def beta_noise_scale_schedule(self, num_sampling_steps):
        """用 Beta 分布 CDF 生成从 max 到 min 的 noise_scale 调度。
        早期（高噪声）noise_scale 大 → 更多随机探索；
        后期（低噪声）noise_scale 小 → 更确定性的精修。"""
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(1 - t, self.noise_scale_alpha, self.noise_scale_beta)
        )
        return (
            self.max_noise_scale
            + (self.min_noise_scale - self.max_noise_scale) * beta_cdf_weights
        )

    def beta_step_scale_schedule(self, num_sampling_steps=None):
        """用 Beta 分布 CDF 生成从 min 到 max 的 step_scale 调度。
        后期步长可以更大（因为噪声已经很小，方向更可靠）。"""
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(t, self.step_scale_alpha, self.step_scale_beta)
        )
        return (
            self.min_step_scale
            + (self.max_step_scale - self.min_step_scale) * beta_cdf_weights
        )

    # ========================================================================
    # 采样过程（推理）
    # ========================================================================
    # 从纯噪声出发，按照噪声调度逐步去噪，生成蛋白质结构。
    # 采样算法基于 EDM 的随机采样器（stochastic sampler），每步包括：
    #   1. 注入少量额外噪声（增加多样性，避免陷入局部最优）
    #   2. 调用去噪网络预测干净坐标
    #   3. Euler 步更新：沿"噪声→干净"方向前进
    #
    # 关键超参数：
    #   α（step_scale）: 步长缩放。越大 → 去噪越激进 → designability ↑, diversity ↓
    #   β（noise_scale）: 噪声注入量。越大 → 随机性越强 → diversity ↑, designability ↓
    def sample(
        self,
        atom_mask,  # Bool['b m'] — 原子掩码（哪些位置有原子）
        num_sampling_steps=None,
        multiplicity=1,       # 每个输入同时生成几个候选结构
        step_scale=None,      # α，可覆盖默认值
        noise_scale=None,     # β，可覆盖默认值
        inference_logging=False,  # 是否显示进度条
        **network_condition_kwargs,
    ):
        # ---- 准备每一步的 α 和 β ----
        # 支持三种策略: 常数、随机选择（训练时）、Beta 分布 CDF
        if self.training and self.step_scale_random is not None:
            # 训练时从预定义列表中随机选一个 step_scale（增加训练多样性）
            step_scales = np.random.choice(self.step_scale_random) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.step_scale_function == "beta":
            step_scales = self.beta_step_scale_schedule(num_sampling_steps)
        else:
            step_scales = default(step_scale, self.step_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        if self.noise_scale_function == "constant":
            noise_scales = default(noise_scale, self.noise_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.noise_scale_function == "beta":
            noise_scales = self.beta_noise_scale_schedule(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid noise scale schedule: {self.noise_scale_function}"
            )
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)  # (B*multiplicity, M, 3)

        # ---- 构建噪声调度 σ 序列 ----
        # 返回 [σ_0, σ_1, ..., σ_N, 0]，从大到小
        if self.sampling_schedule == "af3":
            sigmas = self.sample_schedule_af3(num_sampling_steps)
        elif self.sampling_schedule == "dilated":
            sigmas = self.sample_schedule_dilated(num_sampling_steps)

        # gamma 控制每步注入的额外噪声：
        # 当 σ > gamma_min 时，gamma = gamma_0（注入噪声）
        # 当 σ ≤ gamma_min 时，gamma = 0（不注入，因为噪声已经很小）
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        # 将相邻的 (σ_prev, σ_next, gamma, step_scale, noise_scale) 打包
        sigmas_gammas_ss_ns = list(
            zip(
                sigmas[:-1],    # σ_{t-1}: 当前噪声水平
                sigmas[1:],     # σ_t: 目标噪声水平（更低）
                gammas[1:],
                step_scales,
                noise_scales,
            )
        )

        # ---- 初始化：从纯高斯噪声开始 ----
        # x_0 ~ N(0, σ_max² · I)，方差 = σ_max²，此时结构信息完全被噪声淹没
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        feats = network_condition_kwargs["feats"]

        # ---- 逐步去噪主循环 ----
        coords_traj = [atom_coords]       # 记录每步的坐标轨迹
        x0_coords_traj = []               # 记录每步网络预测的"干净坐标"
        for step_idx, (
            sigma_tm,      # σ_{t-1}: 上一步的噪声水平（较大）
            sigma_t,       # σ_t: 这一步要达到的噪声水平（较小）
            gamma,         # 随机性控制参数
            step_scale,    # α: 当前步的步长缩放
            noise_scale,   # β: 当前步的噪声注入量
        ) in optionally_tqdm(
            enumerate(sigmas_gammas_ss_ns),
            use_tqdm=inference_logging,
            desc="Denoising steps.",
        ):
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            # ---- Step 1: 计算增强后的噪声水平 t̂ ----
            # t̂ = σ_{t-1} * (1 + γ)，γ > 0 时稍微"加点噪声"再去噪
            # 这是 stochastic sampler 的关键：先走一小步"回头路"增加随机性
            t_hat = sigma_tm * (1 + gamma)
            # 额外噪声的方差 = β² * (t̂² - σ_{t-1}²)
            noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)

            # ---- Step 2: 预处理坐标 ----
            # 将坐标中心化到质心（蛋白质结构的平移不变性）
            atom_coords = center(atom_coords, atom_mask)

            # 可选：推理时做随机 SE(3) 增强（旋转+平移）
            # 这使得模型在不同朝向下生成结构，增加多样性
            if self.coordinate_augmentation_inference:
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
                atom_coords = (
                    torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
                )

            # ---- Step 3: 注入额外噪声 ----
            # x̃ = x + β * √(t̂² - σ_{t-1}²) * ε，ε ~ N(0, I)
            eps = noise_scale * sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            # ---- Step 4: 调用去噪网络 ----
            # D_θ(x̃, t̂) → 预测的干净坐标 x̂_0
            with torch.no_grad():
                atom_coords_denoised, net_out = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        **network_condition_kwargs,
                    ),
                )

            # 可选：将含噪坐标刚体对齐到去噪坐标
            # 这有助于在采样过程中保持结构的一致性
            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            # ---- Step 5: Euler 步更新 ----
            # score 估计: d = (x̃ - x̂_0) / t̂（噪声方向的估计）
            # 注意：AF3 论文中可能有错误，使用了 atom_coords 而非 atom_coords_noisy
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            # Euler 更新: x_{next} = x̃ + α * (σ_t - t̂) * d
            # 因为 σ_t < t̂，所以 (σ_t - t̂) < 0，实际是在反方向走（去噪方向）
            # α 控制步长大小：α > 1 时更激进，α < 1 时更保守
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            coords_traj.append(atom_coords_next)
            x0_coords_traj.append(atom_coords_denoised)
            atom_coords = atom_coords_next
        coords_traj.append(atom_coords)

        result = dict(
            sample_atom_coords=atom_coords,     # 最终生成的原子坐标
            coords_traj=coords_traj,            # 完整去噪轨迹（可视化用）
            x0_coords_traj=x0_coords_traj,      # 每步的"干净预测"轨迹
        )

        return result

    # ========================================================================
    # 训练相关
    # ========================================================================

    def loss_weight(self, sigma):
        """损失权重: w(σ) = (σ² + σ_data²) / (σ · σ_data)²
        这个权重使得不同噪声水平下的损失贡献大致均衡。
        σ 很大时 w ≈ 1/σ_data² → 权重较小（高噪声时预测本就不准，不强求）
        σ 很小时 w ≈ 1/σ² → 权重很大（低噪声时应该预测得很准）
        注意: AF3 论文分母中用 +，但 EDM 用 *，作者认为 AF3 可能有误"""
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        """训练时采样噪声水平: σ ~ σ_data · exp(N(P_mean, P_std²))
        对数正态分布的好处：
        - σ 横跨多个数量级（0.0004 ~ 160），对数空间更均匀
        - P_mean=-1.2 → 中位数 σ ≈ σ_data · e^(-1.2) ≈ 4.8
        - 多数训练样本的噪声在"中等"水平，这是网络最难学的区域
        AF3 额外乘以 σ_data（EDM 不乘），等价于偏移 P_mean"""
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts'] — 输入序列特征
        s_trunk,   # Float['b n ts'] — trunk 网络输出
        feats,     # 特征字典（包含 coords, atom_pad_mask 等）
        diffusion_conditioning,  # 预计算的条件信息
        multiplicity=1,
    ):
        """训练前向传播: 加噪 → 去噪 → 返回预测结果（损失在 compute_loss 中计算）"""
        batch_size = feats["coords"].shape[0] // multiplicity
        atom_coords = feats["coords"]  # 真实原子坐标
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        # 中心化 + 随机 SE(3) 增强（旋转+平移），增加训练数据多样性
        # 蛋白质结构的物理性质与绝对位置和朝向无关（等变性）
        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        # ---- 采样噪声水平 σ ----
        if self.synchronize_sigmas:
            # 同一样本的所有 multiplicity 副本使用相同的 σ
            # （公平比较不同候选的去噪能力）
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            # 每个副本独立采样 σ（增加训练多样性）
            sigmas = self.noise_distribution(batch_size * multiplicity)

        # ---- 加噪: x_noisy = x_clean + σ · ε ----
        # 这就是扩散模型的前向过程，极其简单——直接加高斯噪声
        # alphas=1（VP 框架中 α_t 衰减，但 VE/EDM 框架中 α=1，只靠 σ 控制噪声）
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")
        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        # ---- 去噪: D_θ(x_noisy, σ) → x̂_0 ----
        denoised_atom_coords, net_out = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        out_dict = {
            "noised_atom_coords": noised_atom_coords,       # 含噪坐标
            "denoised_atom_coords": denoised_atom_coords,    # 网络预测的干净坐标
            "sigmas": sigmas,                                # 每个样本的噪声水平
            "aligned_true_atom_coords": atom_coords,         # 增强后的真实坐标
        }
        out_dict.update(net_out)

        return out_dict

    # ========================================================================
    # 损失计算
    # ========================================================================
    def compute_loss(
        self,
        feats,
        out_dict,                          # forward() 的输出
        add_smooth_lddt_loss=True,         # 是否加入 smooth lDDT 辅助损失
        add_bond_loss=False,               # 是否加入共价键长损失
        nucleotide_loss_weight=5.0,        # 核酸原子的损失权重倍数
        ligand_loss_weight=10.0,           # 配体原子的损失权重倍数
        fake_atom_weight=1.0,              # 虚拟原子（padding 等）的权重
        residue_type_weight=0.0,           # 残基类型差异化权重
        multiplicity=1,
    ):
        # 关闭 autocast 以确保损失计算在 float32 精度下进行
        # （混合精度训练中，损失计算需要高精度以避免数值不稳定）
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            noised_atom_coords = out_dict["noised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            # ---- 构建原子级掩码和权重 ----
            # resolved_atom_mask: 标记哪些原子在实验结构中有确定坐标
            # （未解析的原子不参与损失计算）
            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()
            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            # fake_atom_weight: 虚拟原子（如 padding 位置）的权重
            # 真实原子权重=1，虚拟原子权重=fake_atom_weight（通常较低）
            fake_atom_mask = feats["fake_atom_mask"]
            fake_atom_weight = (1 - fake_atom_mask) + fake_atom_mask * fake_atom_weight

            # 残基类型权重: 对设计区域中不同氨基酸给予不同权重
            # 例如，稀有氨基酸可能需要更高权重以确保学好
            if residue_type_weight > 0.0:
                # 通过 atom_to_token 矩阵将 token 级信息映射到原子级
                design_atom_mask = torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["design_mask"].float().unsqueeze(-1),
                ).squeeze(-1)
                _res_type_weight = torch.tensor(
                    const.res_type_weight, device=denoised_atom_coords.device
                )
                _res_type_weight = torch.bmm(
                    feats["atom_to_token"].float(),
                    (feats["res_type"].float() @ _res_type_weight)
                    .unsqueeze(-1)
                    .float(),
                ).squeeze(-1)
                # 非设计区域权重=1，设计区域按残基类型加权
                res_type_weight = (
                    1.0 - design_atom_mask
                ) + design_atom_mask * _res_type_weight
                res_type_weight = res_type_weight**residue_type_weight
            else:
                res_type_weight = 1.0

            # ---- 对齐权重: 不同分子类型有不同的损失权重 ----
            align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
            # 确定每个原子属于哪种分子类型（蛋白质/DNA/RNA/配体）
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            # 核酸和配体原子的权重更高：
            # - 核酸 × nucleotide_loss_weight（5x）：核酸原子较少但结构重要
            # - 配体 × ligand_loss_weight（10x）：配体更少但对功能至关重要
            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            # ---- 刚体对齐: 消除平移/旋转差异后再计算 MSE ----
            # 为什么需要对齐？扩散模型生成的结构可能整体平移/旋转了，
            # 但蛋白质结构的质量取决于内部几何关系，与绝对位置无关。
            atom_coords = out_dict["aligned_true_atom_coords"].float()
            if self.mse_rotational_alignment:
                # 完整刚体对齐（Kabsch 算法）：同时消除旋转和平移
                atom_coords_aligned_ground_truth = weighted_rigid_align(
                    atom_coords.detach(),
                    denoised_atom_coords.detach(),
                    align_weights.detach(),
                    mask=feats["atom_resolved_mask"]
                    .float()
                    .repeat_interleave(multiplicity, 0)
                    .detach(),
                )
            else:
                # 仅质心对齐（平移）：更简单，计算更快
                atom_coords_aligned_ground_truth = weighted_rigid_centering(
                    atom_coords,
                    denoised_atom_coords,
                    align_weights,
                    mask=feats["atom_resolved_mask"]
                    .float()
                    .repeat_interleave(multiplicity, 0),
                )

            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # ---- 主损失: 加权 MSE ----
            # L_MSE = w(σ) · Σ_atoms[ weights · ||x̂_0 - x_true||² ] / Σ_atoms[ weights ]
            #
            # 先在 xyz 三个维度上求和（每个原子的欧氏距离平方）
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)  # (B, M) — 每个原子的 MSE
            # 在原子维度上做加权求和，除以加权原子数（归一化）
            # 权重 = 分子类型权重 × 虚拟原子权重 × 残基类型权重 × 解析掩码
            mse_loss = torch.sum(
                mse_loss
                * align_weights
                * fake_atom_weight
                * res_type_weight
                * resolved_atom_mask,
                dim=-1,  # 在原子维度上求和 → (B,)
            ) / (
                torch.sum(
                    3  # 乘 3 是因为分子有 xyz 三个维度，归一化到"每个自由度"的损失
                    * align_weights
                    * fake_atom_weight
                    * res_type_weight
                    * resolved_atom_mask,
                    dim=-1,
                )
                + 1e-5  # 防止除零
            )
            # 按噪声水平加权后取批次平均
            # w(σ) 使低噪声（应该预测更准）的损失权重更大
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            # ---- 辅助损失 1: 共价键长度损失 ----
            # 惩罚预测结构中化学键长度偏离标准值的情况
            # 这是一个硬约束：化学键长有严格的物理范围
            if add_bond_loss:
                bond_loss, num_bonds = compute_bond_loss(
                    pred_atom_coords=out_dict["denoised_atom_coords"].float(),
                    true_coords=atom_coords_aligned_ground_truth,
                    feats=feats,
                )
                total_loss += bond_loss
            else:
                bond_loss = self.zero

            # ---- 辅助损失 2: Smooth lDDT 损失 ----
            # lDDT（Local Distance Difference Test）评估局部结构质量：
            # 对每对近邻原子，比较预测距离和真实距离的差异，
            # 通过 4 个 sigmoid 门槛（0.5, 1.0, 2.0, 4.0 Å）量化精度。
            # 与 MSE 不同，lDDT 是平移/旋转不变的局部指标。
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    # 核酸原子需要特殊处理（它们的局部结构更规则）
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            # 返回总损失和各分项（用于训练监控和日志记录）
            loss_breakdown = {
                "mse_loss": mse_loss,
                "bond_loss": bond_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
