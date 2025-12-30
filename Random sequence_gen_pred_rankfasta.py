# path: tools/dna_screen_keras.py
#!/usr/bin/env python3


from __future__ import annotations
import os
import math
import time
import heapq
from typing import List, Sequence, Tuple, Callable, Optional
import numpy as np

# ========
MODEL_PATH: Optional[str] = "path/to/your_keras_model"  # 例如 "models/my_model.keras" 或 ".h5"；None 则使用占位打分器
OUT_DIR = "results"
TOTAL = 100_000_000          # 每种方法的总序列条数
LENGTH = 80
BATCH = 200_000              # 生成与预测的批大小；根据显存/内存调整
TOPK = 2000
GC = 0.65 #0.72                    # 方法2：链霉菌基因组 GC
SEED = 42
PRED_CHUNK = 50_000          # 预测时的子批大小（BATCH 会被再切分进模型）
DRY_RUN = False              # 置 True 先小规模验证流程



_ASCII = np.array([65, 67, 71, 84], dtype=np.uint8)  # A,C,G,T

def _normalize_probs(p: Sequence[float]) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.shape != (4,) or (arr < 0).any():
        raise ValueError("Probabilities must be 4 non-negative numbers for A,C,G,T.")
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    return arr / s

def _method2_probs_from_gc(gc: float) -> np.ndarray:
    if not (0.0 < gc < 1.0):
        raise ValueError("GC must be in (0,1).")
    at = 1.0 - gc
    return np.array([at/2, gc/2, gc/2, at/2], dtype=float)  # A,C,G,T

def generate_batch(n: int, length: int, probs: np.ndarray, rng: np.random.Generator) -> List[str]:
    idx = rng.choice(4, size=(n, length), p=probs)
    bytes_arr = _ASCII[idx]  # (n, L) uint8
    return [bytes(row).decode("ascii") for row in bytes_arr]

class TopK:
    def __init__(self, k: int):
        self.k = int(k)
        self._heap: List[Tuple[float, int, str]] = []
        self._counter = 0
    def offer(self, score: float, seq: str):
        item = (float(score), self._counter, seq)
        self._counter += 1
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, item)
        else:
            if item[0] > self._heap[0][0]:
                heapq.heapreplace(self._heap, item)
    def extend(self, scores: np.ndarray, seqs: Sequence[str]):
        for s, q in zip(scores.tolist(), seqs):
            self.offer(s, q)
    def sorted_desc(self) -> List[Tuple[float, str]]:
        items = sorted(self._heap, key=lambda t: (t[0], -t[1]), reverse=True)
        return [(s, seq) for (s, _, seq) in items]

def _batch_iter(total: int, batch: int):
    n_batches = math.ceil(total / batch)
    for b in range(n_batches):
        cur = batch if (b + 1) * batch <= total else (total - b * batch)
        yield b, n_batches, cur

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ---- Keras 预测包装 ----
def _load_predict_fn(model_path: Optional[str]) -> Callable[[List[str]], np.ndarray]:
    if model_path is None:
        def _placeholder_predict(seqs: List[str]) -> np.ndarray:
            # 为了流程可跑通：用 GC 比例做分数
            out = np.empty(len(seqs), dtype=np.float32)
            L = len(seqs[0]) if seqs else 1
            for i, s in enumerate(seqs):
                out[i] = (s.count("G") + s.count("C")) / L
            return out
        return _placeholder_predict

    # 延迟导入，避免无 TF 环境时报错
    import tensorflow as tf  # type: ignore

    model = tf.keras.models.load_model(model_path)

    # 约定输入：将 A/C/G/T one-hot 到 (N, L, 4)；如你的模型输入不同，请在此修改。
    _char_to_idx = np.frombuffer(bytearray(256), dtype=np.uint8)
    _char_to_idx[ord('A')] = 0
    _char_to_idx[ord('C')] = 1
    _char_to_idx[ord('G')] = 2
    _char_to_idx[ord('T')] = 3

    def _encode_one_hot(batch_seqs: List[str]) -> np.ndarray:
        n = len(batch_seqs)
        L = len(batch_seqs[0]) if n > 0 else LENGTH
        # uint8 索引矩阵
        idx = np.empty((n, L), dtype=np.uint8)
        for i, s in enumerate(batch_seqs):
            idx[i, :] = _char_to_idx[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        # one-hot
        x = np.zeros((n, L, 4), dtype=np.float32)
        x[np.arange(n)[:, None], np.arange(L)[None, :], idx] = 1.0
        return x

    def predict_fn(seqs: List[str]) -> np.ndarray:
        scores = np.empty(len(seqs), dtype=np.float32)
        # 分块推理，避免显存峰值
        for i in range(0, len(seqs), PRED_CHUNK):
            sub = seqs[i:i+PRED_CHUNK]
            x = _encode_one_hot(sub)
            y = model.predict(x, verbose=0)
            # 假设模型输出形状为 (N, 1) 或 (N,)；若不同，请修改此处取分逻辑
            y = np.asarray(y).reshape(len(sub), -1)
            if y.shape[1] == 1:
                y = y[:, 0]
            else:
                # 若是多输出/多类，这里取第一列或自定义聚合
                y = y[:, 0]
            scores[i:i+len(sub)] = y.astype(np.float32, copy=False)
        return scores

    return predict_fn

def _write_fasta(path: str, scored: List[Tuple[float, str]]):
    with open(path, "w") as f:
        for i, (_score, seq) in enumerate(scored, start=1):
            f.write(f">{i}\n{seq}\n")

def _process_method(tag: str, probs: np.ndarray, total: int, length: int,
                    batch: int, predict_fn: Callable[[List[str]], np.ndarray],
                    seed: int, out_dir: str, topk: int) -> str:
    rng = np.random.default_rng(seed)
    top = TopK(topk)
    _ensure_dir(out_dir)

    t0 = time.time()
    if DRY_RUN:
        total = min(total, batch * 2)

    done = 0
    for bi, n_batches, cur in _batch_iter(total, batch):
        seqs = generate_batch(cur, length, probs, rng)
        scores = predict_fn(seqs)
        if not isinstance(scores, np.ndarray) or scores.shape[0] != len(seqs):
            raise RuntimeError("predict() must return an array of shape (len(seqs),).")
        top.extend(scores, seqs)
        done += cur
        if (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[{tag}] {done:,}/{total:,} ({done/total:.1%}) ~{rate:,.0f}/s heap={len(top._heap)}", flush=True)

    out_fa = os.path.join(out_dir, f"{tag}_top{topk}.fasta")
    _write_fasta(out_fa, top.sorted_desc())
    print(f"[{tag}] wrote {out_fa}")
    return out_fa

def main():
    # 概率设定
    probs1 = _normalize_probs([0.25, 0.25, 0.25, 0.25])           # 方法1：均匀
    probs2 = _normalize_probs(_method2_probs_from_gc(GC))         # 方法2：按 GC=0.72

    predict_fn = _load_predict_fn(MODEL_PATH)

    out1 = _process_method("method1_uniform",  probs1, TOTAL, LENGTH, BATCH, predict_fn, SEED, OUT_DIR, TOPK)
    out2 = _process_method("method2_gc_based", probs2, TOTAL, LENGTH, BATCH, predict_fn, SEED+1, OUT_DIR, TOPK)

    print("Done.")
    print("FASTA outputs:")
    print(" ", out1)
    print(" ", out2)

if __name__ == "__main__":
    main()
