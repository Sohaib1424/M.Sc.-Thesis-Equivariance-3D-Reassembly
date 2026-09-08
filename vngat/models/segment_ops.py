"""
Segment (scatter) primitives, TensorFlow.

`segment_softmax` is the load-bearing one: it lets every attention stage
normalise over a variable-size neighbourhood WITHOUT materialising a dense
(queries x keys) logit matrix. A Breaking Bad scene runs to ~17k vertices and
up to ~86 fragments, so the dense form would allocate a (17k, 8*86) tensor per
head per stage per layer -- three times over, for raw logits, the masked copy
and the softmax.

Every mask involved is block-diagonal (a vertex may only attend to its own
fragment's slots), so forming only the allowed pairs is exact, not an
approximation. `tests/test_segment_ops.py` asserts it matches dense masked
attention numerically.
"""
from __future__ import annotations

import tensorflow as tf

_HALF = (tf.float16, tf.bfloat16)


def at_least_float32(x: tf.Tensor) -> tf.Tensor:
    """
    Promote half precision to float32; leave float32/float64 untouched.

    Used wherever a product must not overflow. An unconditional cast to float32
    would DOWNCAST a float64 model -- which the equivariance tests run in -- and
    silently drop their residual from ~1e-15 to ~1e-8, making a real regression
    indistinguishable from noise.
    """
    return tf.cast(x, tf.float32) if x.dtype in _HALF else x


def _accum_dtype(dtype) -> tf.DType:
    """
    Accumulate in float32 whenever the data is half precision.

    NOT optional. These reductions run over WHOLE FRAGMENTS -- virtual-node
    pooling and the final per-fragment mean sum thousands of values each.
    float16 has ~11 bits of mantissa, so once a running sum reaches ~2048 its
    ulp exceeds 1 and further contributions of order 1 are silently discarded.
    Only the accumulator is widened; results return in the caller's dtype.
    """
    return tf.float32 if dtype in _HALF else dtype


def segment_max(src: tf.Tensor, index: tf.Tensor, num_segments: int) -> tf.Tensor:
    """Per-segment maximum over axis 0. Segments with no members yield 0."""
    acc = _accum_dtype(src.dtype)
    out = tf.math.unsorted_segment_max(tf.cast(src, acc), index, num_segments)
    # unsorted_segment_max fills empty segments with dtype.min; zero them so a
    # later gather cannot pull a huge negative into a live computation.
    floor = tf.constant(-3e38, acc)
    out = tf.where(out > floor, out, tf.zeros_like(out))
    return tf.cast(out, src.dtype)


def segment_sum(src: tf.Tensor, index: tf.Tensor, num_segments: int) -> tf.Tensor:
    """Per-segment sum over axis 0."""
    acc = _accum_dtype(src.dtype)
    return tf.cast(tf.math.unsorted_segment_sum(tf.cast(src, acc), index, num_segments),
                   src.dtype)


def segment_mean(src: tf.Tensor, index: tf.Tensor, num_segments: int) -> tf.Tensor:
    """Per-segment mean over axis 0, with empty segments mapped to 0."""
    total = segment_sum(src, index, num_segments)
    # Integer counts, not a float accumulation: counting thousands of members in
    # float16 hits exactly the limit described in `_accum_dtype`, so the DIVISOR
    # would be wrong too.
    counts = tf.math.unsorted_segment_sum(
        tf.ones_like(index, dtype=tf.int32), index, num_segments)
    counts = tf.cast(tf.maximum(counts, 1), total.dtype)
    shape = [num_segments] + [1] * (len(src.shape) - 1)
    return total / tf.reshape(counts, shape)


def segment_softmax(logits: tf.Tensor, index: tf.Tensor, num_segments: int) -> tf.Tensor:
    """
    Softmax over axis 0 within each segment.

    Every trailing dimension is normalised independently, so a (M, K, H) tensor
    gives one distribution per (slot, head) pair.

    Computed in float32 under mixed precision, for the same reason TF's own
    policy keeps softmax in float32. The logit tensor is one scalar per pair per
    head, so the widened copy is cheap.
    """
    out_dtype = logits.dtype
    work = tf.cast(logits, _accum_dtype(out_dtype))

    # Clamp BEFORE the max-shift. The shift is what makes softmax stable, but it
    # is also what turns a single +inf into NaN: the max becomes inf and
    # inf - inf = NaN, which propagates to every downstream feature. Callers
    # compute scores in float32 so an inf should not arrive; this bounds the
    # damage if one ever does, degrading to a hard argmax instead.
    limit = tf.constant(work.dtype.max / 8, work.dtype)
    work = tf.clip_by_value(work, -limit, limit)

    maxima = segment_max(tf.stop_gradient(work), index, num_segments)
    exp = tf.exp(work - tf.gather(maxima, index))
    denom = tf.gather(segment_sum(exp, index, num_segments), index)
    tiny = tf.constant(1e-30, work.dtype)
    return tf.cast(exp / tf.maximum(denom, tiny), out_dtype)


def blockwise_softmax(logits: tf.Tensor, block_id: tf.Tensor) -> tf.Tensor:
    """
    Softmax over the LAST axis, restricted to entries sharing a block id.

    Only used where the population is small (slots of one scene), so a dense
    within-block matrix is genuinely cheap. Masking with a large negative
    constant rather than -inf keeps a singleton block from producing NaN.
    """
    mask = tf.equal(tf.expand_dims(block_id, 0), tf.expand_dims(block_id, -1))
    while len(mask.shape) < len(logits.shape):
        mask = tf.expand_dims(mask, 1)
    mask = tf.broadcast_to(mask, tf.shape(logits))
    limit = tf.constant(logits.dtype.max / 8, logits.dtype)
    bounded = tf.clip_by_value(logits, -limit, limit)
    neg = tf.constant(logits.dtype.min / 4, logits.dtype)
    return tf.nn.softmax(tf.where(mask, bounded, neg), axis=-1)
