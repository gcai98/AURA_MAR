'''
models/verifier.py

这里放轻量验证器。

建议只放两类模块：

LightweightVerifier(nn.Module)
FastVerifier 或一个 fast_accept_score(...) 函数

其中：

learned verifier 用小 MLP；

fast verifier 直接按你公式里的

𝑣
fast
=
exp
⁡
(
−
𝜌
𝑢
𝑢
−
𝜌
𝑠
𝑠
−
𝜌
𝑐
𝑐
)
v
fast
	​

=exp(−ρ
u
	​

u−ρ
s
	​

s−ρ
c
	​

c)

算。

这样 aura_mar.py 只负责“调度”，而不把 verifier 网络细节全塞进去。
'''
import torch
import torch.nn as nn


def _coerce_tensor(value, reference=None):
    if isinstance(value, torch.Tensor):
        return value
    kwargs = {}
    if isinstance(reference, torch.Tensor):
        kwargs['device'] = reference.device
        kwargs['dtype'] = reference.dtype
    return torch.as_tensor(value, **kwargs)


class LightweightVerifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        num_layers = max(1, int(num_layers))

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


def fast_accept_score(uncertainty, instability=None, inconsistency=None, alpha=1.0, beta=1.0, gamma=1.0):
    uncertainty = _coerce_tensor(uncertainty)
    instability = _coerce_tensor(0.0 if instability is None else instability, reference=uncertainty)
    inconsistency = _coerce_tensor(0.0 if inconsistency is None else inconsistency, reference=uncertainty)
    uncertainty = uncertainty.clamp_min(0.0)
    instability = instability.clamp_min(0.0)
    inconsistency = inconsistency.clamp_min(0.0)
    penalty = alpha * uncertainty + beta * instability + gamma * inconsistency
    return torch.exp(-penalty).clamp(0.0, 1.0)
