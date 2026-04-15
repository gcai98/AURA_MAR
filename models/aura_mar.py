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