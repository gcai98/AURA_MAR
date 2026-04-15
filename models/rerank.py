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