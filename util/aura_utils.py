import torch


def reduce_feature_dim(values):
    if values.ndim <= 2:
        return values
    dims = tuple(range(2, values.ndim))
    return values.mean(dim=dims)


def safe_topk(scores, k, dim=-1):
    if scores.size(dim) == 0:
        empty = scores.narrow(dim, 0, 0)
        return empty, torch.empty_like(empty, dtype=torch.long)

    k = max(0, min(int(k), scores.size(dim)))
    if k == 0:
        empty = scores.narrow(dim, 0, 0)
        return empty, torch.empty_like(empty, dtype=torch.long)
    return torch.topk(scores, k, dim=dim)
