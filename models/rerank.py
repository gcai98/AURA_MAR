'''
models/rerank.py

这里专门放 hardest local windows 和 reranking。

建议内容：

build_local_windows(...)
select_topk_window_centers(...)
generate_local_candidates(...)
score_window_candidates(...)
accept_revise_by_margin(...)

这个文件单独拆出来的好处是：你之后如果发现 reranking 太重，或者想把 window 从方窗改成邻域图，不会动到主模型骨架。
'''
import torch

from util.aura_utils import reduce_feature_dim, safe_topk


def build_local_windows(seq_h, seq_w, radius=1, device=None):
    windows = []
    for center in range(seq_h * seq_w):
        row, col = divmod(center, seq_w)
        indices = []
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                cur_row = row + d_row
                cur_col = col + d_col
                if 0 <= cur_row < seq_h and 0 <= cur_col < seq_w:
                    indices.append(cur_row * seq_w + cur_col)
        windows.append(torch.tensor(indices, device=device, dtype=torch.long))
    return windows


def select_topk_window_centers(scores, topk=8):
    if scores.ndim > 1:
        reduce_dims = tuple(range(scores.ndim - 1))
        scores = scores.mean(dim=reduce_dims)
    _, indices = safe_topk(scores, topk, dim=-1)
    return indices


def generate_local_candidates(tokens, centers, windows):
    candidates = []
    if isinstance(centers, torch.Tensor):
        centers = centers.tolist()

    for center in centers:
        window_indices = windows[int(center)].to(tokens.device)
        candidates.append(tokens.index_select(1, window_indices))
    return candidates


def score_window_candidates(candidates, scorer=None):
    scores = []
    for candidate in candidates:
        if scorer is not None:
            score = scorer(candidate)
        else:
            score = reduce_feature_dim(candidate.abs()).mean(dim=-1)
        scores.append(score)
    if not scores:
        return None
    return torch.stack(scores, dim=0)


def accept_revise_by_margin(base_score, new_score, delta=0.0):
    return new_score >= (base_score + delta)
