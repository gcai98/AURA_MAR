'''
models/difficulty.py

这里专门放 difficulty 相关逻辑。

建议内容：

compute_uncertainty(...)
compute_instability(...)
compute_local_inconsistency(...)
compute_cond_inconsistency(...)
compute_difficulty(...)
compute_gating_score(...)

这样你后面做 ablation 会非常方便，因为 difficulty / gating 是你论文的一个主模块。
'''
import torch

from util.aura_utils import reduce_feature_dim, safe_normalize


def compute_uncertainty(values):
    centered = values - values.mean(dim=-1, keepdim=True)
    variance = reduce_feature_dim(centered.pow(2))
    return variance.sqrt()


def compute_instability(values, previous_values=None):
    if previous_values is None:
        return torch.zeros_like(reduce_feature_dim(values))
    delta = values - previous_values
    return reduce_feature_dim(delta.pow(2)).sqrt()


def compute_local_inconsistency(values):
    if values.shape[1] <= 1:
        return torch.zeros_like(reduce_feature_dim(values))
    left = torch.roll(values, shifts=1, dims=1)
    right = torch.roll(values, shifts=-1, dims=1)
    inconsistency = 0.5 * (values - left).pow(2) + 0.5 * (values - right).pow(2)
    return reduce_feature_dim(inconsistency).sqrt()


def compute_cond_inconsistency(values, reference_values=None):
    if reference_values is None:
        return torch.zeros_like(reduce_feature_dim(values))
    return reduce_feature_dim((values - reference_values).pow(2)).sqrt()


def compute_difficulty(
    uncertainty=None,
    instability=None,
    local_inconsistency=None,
    cond_inconsistency=None,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    components = []
    if uncertainty is not None:
        components.append(alpha * safe_normalize(uncertainty))
    if instability is not None:
        components.append(beta * safe_normalize(instability))
    if local_inconsistency is not None:
        components.append(gamma * safe_normalize(local_inconsistency))
    if cond_inconsistency is not None:
        components.append(gamma * safe_normalize(cond_inconsistency))

    if not components:
        raise ValueError("At least one difficulty component must be provided.")

    total = components[0]
    for component in components[1:]:
        total = total + component
    return safe_normalize(total)


def compute_gating_score(difficulty, accept_score=None, gate_tau=0.5, uncertainty=None):
    if accept_score is None:
        accept_score = torch.ones_like(difficulty)
    if uncertainty is None:
        uncertainty = torch.zeros_like(difficulty)
    return accept_score - difficulty - 0.5 * safe_normalize(uncertainty) - gate_tau
