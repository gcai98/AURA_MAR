'''
models/aura_mar.py

这是最核心的文件，建议你直接从 models/mar.py 复制一份开始改，不要一开始用继承硬套。

原因很简单：原版 MAR 的 sample_tokens() 已经把整套 token 生成流程写在一个函数里了，AURA-MAR 的 action decision 正好要深入这个循环内部；硬继承会变得比复制重构更绕。

这个文件里你要做的事：

保留原版：
patchify / unpatchify
sample_orders
forward_mae_encoder
forward_mae_decoder
新增或改造：
forward_backbone_step(...)
compute_uncertainty(...)
compute_instability(...)
compute_inconsistency(...)
compute_difficulty(...)
build_verifier_inputs(...)
decide_actions(...)
select_hard_windows(...)
apply_local_reranking(...)
sample_tokens_aura(...)
末尾新增工厂函数：
aura_mar_base
aura_mar_large
aura_mar_huge

也就是说，这个文件以后会成为你的主模型文件。
'''
from functools import partial

import torch
import torch.nn as nn

from models.difficulty import (
    compute_cond_inconsistency,
    compute_difficulty as combine_difficulty,
    compute_instability,
    compute_local_inconsistency,
    compute_uncertainty,
)
from models.mar import MAR
from models.rerank import (
    build_local_windows,
    generate_local_candidates,
    score_window_candidates,
    select_topk_window_centers,
)
from models.verifier import LightweightVerifier, fast_accept_score


class AURAMAR(MAR):
    def __init__(
        self,
        use_aura_sampling=False,
        return_loss_dict=False,
        verifier_hidden_dim=128,
        verifier_num_layers=2,
        gate_tau=0.5,
        difficulty_alpha=1.0,
        difficulty_beta=1.0,
        difficulty_gamma=1.0,
        tau_d_low=0.25,
        tau_d_high=0.75,
        tau_v_high=0.8,
        tau_v_low=0.2,
        tau_v_keep=0.8,
        tau_v_revise=0.2,
        tau_drift=0.0,
        window_radius=1,
        topk_windows=4,
        rerank_K=1,
        delta_V=0.0,
        lambda_ver=0.0,
        lambda_rank=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_aura_sampling = use_aura_sampling
        self.return_loss_dict = return_loss_dict
        self.gate_tau = gate_tau
        self.difficulty_alpha = difficulty_alpha
        self.difficulty_beta = difficulty_beta
        self.difficulty_gamma = difficulty_gamma
        self.tau_d_low = tau_d_low
        self.tau_d_high = tau_d_high
        self.tau_v_high = tau_v_high
        self.tau_v_low = tau_v_low
        self.tau_v_keep = tau_v_keep
        self.tau_v_revise = tau_v_revise
        self.tau_drift = tau_drift
        self.window_radius = window_radius
        self.topk_windows = topk_windows
        self.rerank_K = rerank_K
        self.delta_V = delta_V
        self.lambda_ver = lambda_ver
        self.lambda_rank = lambda_rank
        self.verifier = LightweightVerifier(
            input_dim=4,
            hidden_dim=verifier_hidden_dim,
            num_layers=verifier_num_layers,
        )

    def compute_difficulty(self, token_state, previous_state=None, reference_state=None):
        uncertainty = compute_uncertainty(token_state)
        instability = compute_instability(token_state, previous_state)
        local_inconsistency = compute_local_inconsistency(token_state)
        cond_inconsistency = compute_cond_inconsistency(token_state, reference_state)
        difficulty = combine_difficulty(
            uncertainty=uncertainty,
            instability=instability,
            local_inconsistency=local_inconsistency,
            cond_inconsistency=cond_inconsistency,
            alpha=self.difficulty_alpha,
            beta=self.difficulty_beta,
            gamma=self.difficulty_gamma,
        )
        return {
            'difficulty': difficulty,
            'uncertainty': uncertainty,
            'instability': instability,
            'local_inconsistency': local_inconsistency,
            'cond_inconsistency': cond_inconsistency,
        }

    def build_verifier_inputs(self, token_state, difficulty_pack, mask=None):
        del token_state
        difficulty = difficulty_pack['difficulty']
        if mask is None:
            mask_feature = torch.zeros_like(difficulty)
        else:
            mask_feature = mask.to(difficulty.dtype)

        return torch.stack(
            [
                difficulty,
                difficulty_pack['uncertainty'],
                difficulty_pack['instability'],
                mask_feature,
            ],
            dim=-1,
        )

    def decide_actions(self, difficulty_pack, verifier_scores=None):
        difficulty = difficulty_pack['difficulty']
        if verifier_scores is None:
            verifier_scores = fast_accept_score(
                difficulty_pack['uncertainty'],
                difficulty_pack['instability'],
                difficulty_pack['local_inconsistency'],
            )

        easy = difficulty <= self.tau_d_low
        hard = difficulty >= self.tau_d_high
        keep = easy | (verifier_scores >= self.tau_v_keep)
        revise = hard & (verifier_scores <= self.tau_v_revise)
        accept = ~(keep | revise)
        gated = difficulty >= self.gate_tau

        return {
            'keep': keep,
            'accept': accept,
            'revise': revise,
            'gated': gated,
            'verifier_scores': verifier_scores,
        }

    def select_hard_windows(self, difficulty):
        centers = select_topk_window_centers(difficulty, topk=self.topk_windows)
        windows = build_local_windows(
            seq_h=self.seq_h,
            seq_w=self.seq_w,
            radius=self.window_radius,
            device=difficulty.device,
        )
        return centers, windows

    def apply_local_reranking(self, tokens, difficulty=None):
        if difficulty is None or self.topk_windows <= 0 or self.rerank_K <= 0:
            return tokens

        centers, windows = self.select_hard_windows(difficulty)
        candidates = generate_local_candidates(tokens, centers, windows)
        if not candidates:
            return tokens

        _ = score_window_candidates(candidates[:self.rerank_K])
        return tokens

    def sample_tokens_aura(self, *args, **kwargs):
        return super().sample_tokens(*args, **kwargs)

    def forward(self, imgs, labels):
        loss_mar = super().forward(imgs, labels)
        if not self.return_loss_dict:
            return loss_mar

        with torch.no_grad():
            token_state = self.patchify(imgs)
            difficulty_pack = self.compute_difficulty(token_state)
            verifier_inputs = self.build_verifier_inputs(token_state, difficulty_pack)
            verifier_scores = torch.sigmoid(
                self.verifier(verifier_inputs.reshape(-1, verifier_inputs.shape[-1]))
            ).reshape_as(difficulty_pack['difficulty'])
            fast_scores = fast_accept_score(
                difficulty_pack['uncertainty'],
                difficulty_pack['instability'],
                difficulty_pack['local_inconsistency'],
            )

        loss_ver = loss_mar.new_zeros(())
        loss_rank = loss_mar.new_zeros(())
        return {
            'loss_total': loss_mar,
            'loss_mar': loss_mar,
            'loss_ver': loss_ver,
            'loss_rank': loss_rank,
            'metrics': {
                'difficulty_mean': difficulty_pack['difficulty'].mean(),
                'verifier_mean': verifier_scores.mean(),
                'fast_accept_mean': fast_scores.mean(),
            },
        }

    def sample_tokens(self, *args, **kwargs):
        if self.use_aura_sampling:
            return self.sample_tokens_aura(*args, **kwargs)
        return super().sample_tokens(*args, **kwargs)


def aura_mar_base(**kwargs):
    model = AURAMAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def aura_mar_large(**kwargs):
    model = AURAMAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def aura_mar_huge(**kwargs):
    model = AURAMAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
