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