"""
The dominant VRAM cost is VirtualNodeCommunicationBlock's dense masked
attention:  logits of shape (F*K, H, N) and (N, H, F*K), then masked_fill,
then softmax -- three tensors of that size, retained for backward.

But every virtual node only ever attends to ITS OWN fragment's vertices (the
mask enforces exactly that). So the dense form computes F*K*N scores to then
throw away all but K*N of them: an F-fold blowup.

This validates that a segment ("gathered") formulation is NUMERICALLY
IDENTICAL to the dense masked one, for both Stage 1 (upward) and Stage 3
(downward), and that both are SO(3)-equivariant.
"""
import numpy as np
rng = np.random.default_rng(1)

def rand_rot(rng):
    U, _, Vt = np.linalg.svd(rng.normal(size=(3, 3)))
    R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R

def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

# --- scene: F fragments, K slots each, N vertices, H heads, C head-channels --
F, K, H, C = 4, 8, 3, 5
frag_sizes = [7, 11, 4, 9]
N = sum(frag_sizes)
frag_id = np.concatenate([np.full(n, f) for f, n in enumerate(frag_sizes)])
scale = (C * 3) ** -0.5

Wq = rng.normal(size=(H * C, C)); Wk = rng.normal(size=(H * C, C)); Wv = rng.normal(size=(H * C, C))

def vnlin(W, x):                       # x: (n, C_in, 3) -> (n, C_out, 3)
    return np.einsum('oi,nid->nod', W, x)

# =============== STAGE 1: upward pooling (vnodes attend over vertices) =======
def stage1_dense(qry, xv):
    """qry: (F*K, C, 3) queries; xv: (N, C, 3) vertices. The current code."""
    Q = vnlin(Wq, qry).reshape(F * K, H, C, 3)
    Kk = vnlin(Wk, xv).reshape(N, H, C, 3)
    V = vnlin(Wv, xv).reshape(N, H, C, 3)
    logits = np.einsum('qhoc,khoc->qhk', Q, Kk) * scale       # (F*K, H, N)  <-- BIG
    vnode_frag = np.repeat(np.arange(F), K)
    mask = vnode_frag[:, None] == frag_id[None, :]            # (F*K, N)     <-- BIG
    logits = np.where(mask[:, None, :], logits, -np.inf)      #              <-- BIG
    a = softmax(logits, axis=-1)
    return np.einsum('qhk,khoc->qhoc', a, V).reshape(F * K, H * C, 3)

def stage1_segment(qry, xv):
    """Same result, memory O(N*H*K) instead of O(N*H*F*K)."""
    Q = vnlin(Wq, qry).reshape(F, K, H, C, 3)
    Kk = vnlin(Wk, xv).reshape(N, H, C, 3)
    V = vnlin(Wv, xv).reshape(N, H, C, 3)
    # each vertex scores only against the K slots of its OWN fragment
    logits = np.einsum('nkhoc,nhoc->nhk', Q[frag_id], Kk) * scale   # (N, H, K)  <-- small
    # softmax over the vertices WITHIN each fragment, per (h, k): scatter-softmax
    a = np.zeros_like(logits)
    for f in range(F):
        m = frag_id == f
        a[m] = softmax(logits[m], axis=0)
    out = np.zeros((F, K, H, C, 3))
    for k in range(K):                       # loop K (=8), never materialise (N,H,K,C,3)
        np.add.at(out[:, k], frag_id, a[:, :, k][:, :, None, None] * V)
    return out.reshape(F * K, H * C, 3)

# =============== STAGE 3: downward broadcast (vertices attend over vnodes) ===
def stage3_dense(xv, vn):
    Q = vnlin(Wq, xv).reshape(N, H, C, 3)
    Kk = vnlin(Wk, vn).reshape(F * K, H, C, 3)
    V = vnlin(Wv, vn).reshape(F * K, H, C, 3)
    logits = np.einsum('qhoc,khoc->qhk', Q, Kk) * scale       # (N, H, F*K)  <-- BIG
    vnode_frag = np.repeat(np.arange(F), K)
    mask = frag_id[:, None] == vnode_frag[None, :]
    logits = np.where(mask[:, None, :], logits, -np.inf)
    a = softmax(logits, axis=-1)
    return np.einsum('qhk,khoc->qhoc', a, V).reshape(N, H * C, 3)

def stage3_segment(xv, vn):
    Q = vnlin(Wq, xv).reshape(N, H, C, 3)
    Kk = vnlin(Wk, vn).reshape(F, K, H, C, 3)
    V = vnlin(Wv, vn).reshape(F, K, H, C, 3)
    logits = np.einsum('nhoc,nkhoc->nhk', Q, Kk[frag_id]) * scale   # (N, H, K) <-- small
    a = softmax(logits, axis=-1)                                    # plain local softmax
    out = np.zeros((N, H, C, 3))
    for k in range(K):
        out += a[:, :, k][:, :, None, None] * V[frag_id][:, k]
    return out.reshape(N, H * C, 3)

qry = rng.normal(size=(F * K, C, 3))
xv  = rng.normal(size=(N, C, 3))
vn  = rng.normal(size=(F * K, C, 3))

d1, s1 = stage1_dense(qry, xv), stage1_segment(qry, xv)
d3, s3 = stage3_dense(xv, vn),  stage3_segment(xv, vn)
print(f"[1] stage1 dense vs segment   max|diff| = {np.abs(d1-s1).max():.3e}")
print(f"[2] stage3 dense vs segment   max|diff| = {np.abs(d3-s3).max():.3e}")

# --- equivariance of the segment versions -----------------------------------
R = rand_rot(rng)
rot = lambda t: np.einsum('ij,...j->...i', R, t)
print(f"[3] stage1 segment equivariant  max|diff| = "
      f"{np.abs(stage1_segment(rot(qry), rot(xv)) - rot(s1)).max():.3e}")
print(f"[4] stage3 segment equivariant  max|diff| = "
      f"{np.abs(stage3_segment(rot(xv), rot(vn)) - rot(s3)).max():.3e}")

# --- memory accounting at realistic training scale ---------------------------
print("\nPeak score-tensor elements at a realistic scale (N=200k verts, F=20, K=8, H=4):")
Nr, Fr, Kr, Hr = 200_000, 20, 8, 4
dense = Fr * Kr * Hr * Nr
seg   = Nr * Hr * Kr
print(f"  dense  (F*K, H, N) = {dense:,} elts = {dense*4/2**30:6.2f} GiB per tensor (x3: mask, fill, softmax)")
print(f"  segment (N, H, K)  = {seg:,} elts = {seg*4/2**30:6.2f} GiB per tensor")
print(f"  reduction factor   = {dense/seg:.0f}x")
