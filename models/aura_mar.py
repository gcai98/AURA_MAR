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
import math
from functools import partial

import torch
import torch.nn as nn

from models.difficulty import (
    compute_cond_inconsistency,
    compute_difficulty as combine_difficulty,
    compute_gating_score,
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
from util.aura_utils import masked_mean


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
        self.last_aura_stats = []

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

    def build_verifier_inputs(self, token_state, difficulty_pack, mask=None, step_ratio=None):
        token_norm = token_state.pow(2).mean(dim=-1).sqrt()
        difficulty = difficulty_pack['difficulty']
        if mask is None:
            mask_feature = torch.zeros_like(difficulty)
        else:
            mask_feature = mask.to(difficulty.dtype)
        if step_ratio is None:
            step_feature = torch.zeros_like(difficulty)
        else:
            step_feature = torch.full_like(difficulty, float(step_ratio))

        return torch.stack(
            [
                difficulty,
                difficulty_pack['uncertainty'],
                0.5 * (difficulty_pack['instability'] + difficulty_pack['local_inconsistency']),
                0.5 * mask_feature + 0.5 * step_feature + 0.1 * token_norm,
            ],
            dim=-1,
        )

    def decide_actions(self, difficulty_pack, verifier_scores=None, active_mask=None):
        difficulty = difficulty_pack['difficulty']
        if verifier_scores is None:
            verifier_scores = fast_accept_score(
                difficulty_pack['uncertainty'],
                difficulty_pack['instability'],
                difficulty_pack['local_inconsistency'],
            )
        gating_score = compute_gating_score(
            difficulty,
            accept_score=verifier_scores,
            gate_tau=self.gate_tau,
            uncertainty=difficulty_pack['uncertainty'],
        )

        easy = difficulty <= self.tau_d_low
        hard = difficulty >= self.tau_d_high
        low_uncertainty = difficulty_pack['uncertainty'] <= self.tau_d_low
        high_uncertainty = difficulty_pack['uncertainty'] >= self.tau_d_high
        strong_accept = verifier_scores >= self.tau_v_keep
        weak_accept = verifier_scores >= self.tau_v_high
        weak_reject = verifier_scores <= self.tau_v_low
        strong_reject = verifier_scores <= self.tau_v_revise

        keep = easy & low_uncertainty & strong_accept
        revise = hard & high_uncertainty & strong_reject
        accept = (~keep) & (~revise) & (weak_accept | (~weak_reject))
        gated = gating_score >= 0

        if active_mask is not None:
            active_mask = active_mask.to(torch.bool)
            keep = keep & active_mask
            revise = revise & active_mask
            accept = accept & active_mask
            gated = gated & active_mask

        fallback_accept = active_mask if active_mask is not None else torch.ones_like(accept, dtype=torch.bool)
        unresolved = ~(keep | revise | accept)
        accept = accept | (unresolved & fallback_accept)

        return {
            'keep': keep,
            'accept': accept,
            'revise': revise,
            'gated': gated,
            'gating_score': gating_score,
            'verifier_scores': verifier_scores,
        }

    def summarize_step_stats(self, step, num_iter, difficulty_pack, action_pack, active_mask):
        return {
            'step': float(step),
            'step_ratio': float(step + 1) / float(max(num_iter, 1)),
            'difficulty_mean': float(masked_mean(difficulty_pack['difficulty'], active_mask)),
            'uncertainty_mean': float(masked_mean(difficulty_pack['uncertainty'], active_mask)),
            'instability_mean': float(masked_mean(difficulty_pack['instability'], active_mask)),
            'local_inconsistency_mean': float(masked_mean(difficulty_pack['local_inconsistency'], active_mask)),
            'verifier_mean': float(masked_mean(action_pack['verifier_scores'], active_mask)),
            'gating_mean': float(masked_mean(action_pack['gating_score'], active_mask)),
            'keep_ratio': float(masked_mean(action_pack['keep'].to(difficulty_pack['difficulty'].dtype), active_mask)),
            'accept_ratio': float(masked_mean(action_pack['accept'].to(difficulty_pack['difficulty'].dtype), active_mask)),
            'revise_ratio': float(masked_mean(action_pack['revise'].to(difficulty_pack['difficulty'].dtype), active_mask)),
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

    def forward_backbone_step(self, tokens, mask, class_embedding):
        x = self.forward_mae_encoder(tokens, mask, class_embedding)
        z = self.forward_mae_decoder(x, mask)
        return z

    def _compute_next_mask(self, step, num_iter, mask, orders, bsz):
        mask_ratio = torch.tensor(
            [math.cos(math.pi / 2.0 * (step + 1) / num_iter)],
            device=mask.device,
        )
        mask_len = torch.floor(mask_ratio * self.seq_len)
        mask_len = torch.maximum(
            torch.ones_like(mask_len),
            torch.minimum(mask.sum(dim=-1, keepdim=True) - 1, mask_len),
        )
        mask_len_value = int(mask_len[0].item())
        mask_next = torch.zeros(bsz, self.seq_len, device=mask.device)
        mask_next = torch.scatter(
            mask_next,
            dim=-1,
            index=orders[:, :mask_len_value],
            src=torch.ones_like(mask_next),
        ).bool()
        return mask_len, mask_next

    def sample_tokens_aura(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)
        previous_tokens = None
        self.last_aura_stats = []

        indices = list(range(num_iter))
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices)

        for step in indices:
            cur_tokens = tokens.clone()

            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            model_tokens = tokens
            model_mask = mask
            if not cfg == 1.0:
                model_tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                model_mask = torch.cat([mask, mask], dim=0)

            z_full = self.forward_backbone_step(model_tokens, model_mask, class_embedding)
            mask_len, mask_next = self._compute_next_mask(step, num_iter, mask, orders, bsz)

            if step >= num_iter - 1:
                mask_to_pred = mask.bool()
            else:
                mask_to_pred = torch.logical_xor(mask.bool(), mask_next.bool())

            sample_mask = mask_to_pred
            if not cfg == 1.0:
                sample_mask = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            z_samples = z_full[sample_mask.nonzero(as_tuple=True)]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            sampled_token_latent = self.diffloss.sample(z_samples, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent

            active_mask = mask_to_pred
            difficulty_pack = self.compute_difficulty(
                cur_tokens,
                previous_state=previous_tokens,
                reference_state=tokens,
            )
            verifier_inputs = self.build_verifier_inputs(
                cur_tokens,
                difficulty_pack,
                mask=active_mask,
                step_ratio=float(step + 1) / float(max(num_iter, 1)),
            )
            verifier_scores = torch.sigmoid(
                self.verifier(verifier_inputs.reshape(-1, verifier_inputs.shape[-1]))
            ).reshape_as(difficulty_pack['difficulty'])
            fast_scores = fast_accept_score(
                difficulty_pack['uncertainty'],
                difficulty_pack['instability'],
                difficulty_pack['local_inconsistency'],
                alpha=self.difficulty_alpha,
                beta=self.difficulty_beta,
                gamma=self.difficulty_gamma,
            )
            combined_scores = 0.5 * verifier_scores + 0.5 * fast_scores
            action_pack = self.decide_actions(
                difficulty_pack,
                verifier_scores=combined_scores,
                active_mask=active_mask,
            )

            keep_mask = action_pack['keep']
            revise_mask = action_pack['revise']
            if previous_tokens is not None:
                cur_tokens[keep_mask.nonzero(as_tuple=True)] = previous_tokens[keep_mask.nonzero(as_tuple=True)]
                cur_tokens[revise_mask.nonzero(as_tuple=True)] = tokens[revise_mask.nonzero(as_tuple=True)]
            cur_tokens = self.apply_local_reranking(cur_tokens, difficulty=difficulty_pack['difficulty'])

            previous_tokens = tokens.clone()
            tokens = cur_tokens.clone()
            mask = mask_next.float()
            self.last_aura_stats.append(
                self.summarize_step_stats(step, num_iter, difficulty_pack, action_pack, active_mask)
            )

        tokens = self.unpatchify(tokens)
        return tokens

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
                'gating_mean': compute_gating_score(
                    difficulty_pack['difficulty'],
                    accept_score=fast_scores,
                    gate_tau=self.gate_tau,
                    uncertainty=difficulty_pack['uncertainty'],
                ).mean(),
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
