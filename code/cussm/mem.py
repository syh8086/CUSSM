# -*- coding: utf-8 -*-
"""分块预算：让峰值内存与查询数解耦（2026-09-29，为消除云端 OOM 而建）。

## 背景

2026-09-29 云端两条链被内核 OOM 杀掉（`dmesg`：Track A′ anon-rss 28,513,032 kB ≈ 27.2 GiB；
度量 anon-rss 27,895,100 kB ≈ 26.6 GiB）。静态账本（`.workbuddy/mem_ledger.py`）定位到
三处**在超参搜索期对全量 `val`（或全量 `test`）物化 `(n_query, N_R)`** 的地方：

| 位置 | 改造前（算术和，上界） | 改造后 |
|---|---|---|
| `code/baselines/models.py::LinFuse.fit` | fr_en 8.3 GB ～ Track A′ 119.9 GB | 与 n_val 解耦 |
| `code/cussm/model.py::CUSSM._tune`（主凶，同时持有 8 组同形状中间量） | yago3_10 43.5 GB、Track A′ 119.9 GB | 同上 |
| `code/measure_ch5.py` ③ 路由分解段 | yago3_10 11.0 GB | 同上 |

改为**按行分块**后，峰值 = `chunk × N_R × itemsize × 块内活跃数组数`，与查询数无关。

## 为什么分块不改变数值

被分块的三类运算都是**逐行独立**的：

* `_zscore` / `_softmax_rows` / `_rownorm` 都走 `axis=1`，行与行之间无耦合；
* `np.argmax(S, axis=1)` 只看该行；
* 命中计数用**整数**累加（整数加法满足结合律 ⇒ 逐块累加与整块一次算完全相等），
  且**只在最后做一次除法**（若逐块 `.mean()` 再相加，会引入与整块不同的浮点舍入）。

唯一的理论风险是 `(n, d) @ (d, N)` 矩阵乘：BLAS 可能因 `n` 不同而选择不同的分块
策略，使**同一行**的末位出现 1 ulp 差异。这一点不由推理判定，而由
`.workbuddy/ab_chunk_parity.py` 在真实数据上逐位断言。
"""
from __future__ import annotations

import os

DEFAULT_CHUNK = 512


def budget_bytes() -> float | None:
    """块内存预算（字节）。

    读环境变量 `CUSSM_BLOCK_BUDGET_MB`（正整数，单位 MB）。**未设时返回 None**，
    意为"只受调用方传入的 `chunk` 约束"——这样默认行为与改造前的 `chunk=512`
    完全一致，不引入任何新的超参；需要更紧的约束时（例如候选池被换成百万级）
    再显式设该变量，无需改代码。
    """
    v = os.environ.get("CUSSM_BLOCK_BUDGET_MB")
    if not v:
        return None
    try:
        mb = float(v)
    except ValueError:
        return None
    return mb * 2 ** 20 if mb > 0 else None


def plan_block(n_cols: int, arrays: int = 4, itemsize: int = 4,
               chunk: int = DEFAULT_CHUNK) -> int:
    """返回**不超过内存预算**的块行数。

    `arrays`：该点块内**同时活跃**的 `(m, n_cols)` 数组个数（由调用点在注释中列明）。
    """
    chunk = int(max(1, chunk))
    n_cols = int(max(1, n_cols))
    b = budget_bytes()
    if b is None:
        return chunk
    per_row = n_cols * itemsize * max(1, int(arrays))
    return int(max(1, min(chunk, int(b // per_row))))


def block_slices(n: int, chunk: int):
    """按行切块，返回**连续**行区间。

    连续性是有意的：它保证各块的拼接顺序等于原始行序，于是
    `argmax` 的按行结果与整数命中数的累加顺序都与整块一次算相同。
    """
    step = int(max(1, chunk))
    for i in range(0, int(n), step):
        yield slice(i, min(i + step, int(n)))
