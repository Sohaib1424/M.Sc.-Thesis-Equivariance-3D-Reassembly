"""
Vector Neuron primitives, TensorFlow.

Features are "vector lists": (..., C, 3) -- C channels, each a 3-vector. The
unifying rule (Deng et al., 2021): NEVER LET A LEARNED WEIGHT TOUCH THE
TRAILING XYZ AXIS. Every op here either mixes across the CHANNEL axis only
(which commutes with rotation, since rotation acts on xyz), or uses inner
products between co-rotating vectors (which are invariant) to decide how to
combine them.
"""
from __future__ import annotations

import tensorflow as tf

from .segment_ops import at_least_float32

EPS = 1e-6


def _at_least_float32(x):
    """Alias, so this module's callers need only one import. Defined once in
    segment_ops so the rule cannot drift between files."""
    return at_least_float32(x)


class VNLinear(tf.keras.layers.Layer):
    """
    (..., C_in, 3) -> (..., C_out, 3). Channel mixing only.

    VNLinear(Rx) == R VNLinear(x) for every R in SO(3), because R acts on the
    xyz axis and W on the channel axis, and the two commute trivially.
    """

    def __init__(self, out_channels: int, use_bias: bool = False, **kw):
        super().__init__(**kw)
        self.out_channels = out_channels
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = int(input_shape[-2])
        # Match torch.nn.Linear's default init so the two ports start from
        # comparable scales.
        bound = 1.0 / max(in_channels, 1) ** 0.5
        self.w = self.add_weight(
            name="w", shape=(in_channels, self.out_channels), dtype=self.dtype,
            initializer=tf.keras.initializers.RandomUniform(-bound, bound), trainable=True)
        self.b = self.add_weight(
            name="b", shape=(self.out_channels,), dtype=self.dtype,
            initializer=tf.keras.initializers.RandomUniform(-bound, bound),
            trainable=True) if self.use_bias else None
        super().build(input_shape)

    def call(self, x):
        # transpose so the channel axis is last, mix, transpose back
        out = tf.linalg.matmul(tf.linalg.matrix_transpose(x), tf.cast(self.w, x.dtype))
        if self.b is not None:
            out = out + tf.cast(self.b, x.dtype)
        return tf.linalg.matrix_transpose(out)


class VNLeakyReLU(tf.keras.layers.Layer):
    """
    Equivariant nonlinearity.

    For each channel a learned direction q = VNLinear(x) (itself equivariant);
    the component of x along q is attenuated only when <x, q> < 0. Equivariant
    because the branch decision <x, q> is a rotation-invariant scalar -- it
    never consults an externally fixed frame.
    """

    def __init__(self, negative_slope: float = 0.2, share_nonlinearity: bool = False, **kw):
        super().__init__(**kw)
        self.negative_slope = negative_slope
        self.share = share_nonlinearity

    def build(self, input_shape):
        c = int(input_shape[-2])
        self.map_to_dir = VNLinear(1 if self.share else c, dtype=self.dtype)
        super().build(input_shape)

    def call(self, x):
        q = self.map_to_dir(x)
        if q.shape[-2] != x.shape[-2]:
            q = tf.broadcast_to(q, tf.shape(x))
        # float32 for the inner products. These are elementwise products, which
        # mixed-precision policies do NOT promote, so in float16 any activation
        # above ~256 overflows to inf here -- before the sum that would have
        # been promoted. dot and q_sq then both become inf and dot/q_sq is NaN.
        xf, qf = _at_least_float32(x), _at_least_float32(q)
        dot = tf.reduce_sum(xf * qf, axis=-1, keepdims=True)
        q_sq = tf.maximum(tf.reduce_sum(qf * qf, axis=-1, keepdims=True),
                          tf.constant(EPS, xf.dtype))
        proj = (dot / q_sq) * qf
        out = tf.where(dot >= 0, xf, xf - (1 - self.negative_slope) * proj)
        return tf.cast(out, x.dtype)


class VNLayerNorm(tf.keras.layers.Layer):
    """
    Equivariant normalisation with NO batch statistics.

    Rotation changes a vector's direction, never its length, so the per-channel
    norm ||x_c|| is an invariant scalar. This normalises those norms ACROSS
    CHANNELS within each node independently, then rescales the directions --
    directions are never touched, so equivariance holds.

    Default rather than batch norm because the "batch" here is a variable-size
    set of mesh vertices whose count swings by more than an order of magnitude
    between scenes, and the replicas see different scenes. Batch statistics are
    noisy, replica-dependent and mismatched between train and eval; per-node
    normalisation removes the problem rather than synchronising it.
    """

    def __init__(self, eps: float = 1e-5, **kw):
        super().__init__(**kw)
        self.eps = eps

    def build(self, input_shape):
        self.gain = self.add_weight(
            name="gain", shape=(int(input_shape[-2]),), dtype=self.dtype,
            initializer="ones", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # float32 for the norm and its square. In half precision a channel norm
        # above ~256 squares to inf, the RMS becomes inf, and the scale collapses
        # to zero -- silently ZEROING the features rather than normalising them.
        # That is not NaN, so nothing downstream would flag it.
        xf = _at_least_float32(x)
        norm = tf.sqrt(tf.reduce_sum(xf * xf, axis=-1) + 1e-12)
        rms = tf.sqrt(tf.maximum(tf.reduce_mean(norm ** 2, axis=-1, keepdims=True),
                                 tf.constant(self.eps, xf.dtype)))
        scale = tf.expand_dims(tf.cast(self.gain, xf.dtype) / rms, -1)
        return tf.cast(xf * scale, x.dtype)


class VNBatchNorm(tf.keras.layers.Layer):
    """Equivariant batch norm on the invariant per-channel norms. Kept for
    comparison; note it adds running statistics estimated from whatever scenes a
    replica happened to draw."""

    def __init__(self, **kw):
        super().__init__(**kw)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(dtype=self.dtype)
        super().build(input_shape)

    def call(self, x, training=False):
        shape = tf.shape(x)
        flat = tf.reshape(_at_least_float32(x), [-1, x.shape[-2], 3])
        norm = tf.sqrt(tf.reduce_sum(flat * flat, axis=-1) + 1e-12) + EPS
        out = flat / tf.expand_dims(norm, -1) * tf.expand_dims(self.bn(norm, training=training), -1)
        return tf.cast(tf.reshape(out, shape), x.dtype)


def make_norm(kind: str, **kw):
    kind = (kind or "layer").lower()
    if kind == "layer":
        return VNLayerNorm(**kw)
    if kind == "batch":
        return VNBatchNorm(**kw)
    if kind == "none":
        return tf.keras.layers.Layer()
    raise ValueError(f"norm must be 'layer', 'batch' or 'none', got {kind!r}")


class VNInvariant(tf.keras.layers.Layer):
    """
    Equivariant vectors -> rotation-invariant scalars, via the pairwise Gram
    matrix G_ij = <x_i, x_j> (invariant: both vectors co-rotate and rotations
    preserve inner products), flattened.

    `bottleneck` is a mandatory memory control, not a nicety. The Gram matrix is
    O(C^2) PER ENTITY: at hidden=64 the edge head's input has 131 channels, so a
    batch with ~95k edges would allocate 95k x 131 x 131 floats -- 6.5 GB in one
    tensor. Projecting to `bottleneck` channels first (equivariantly, with a
    VNLinear, so invariance is exact) makes it ~100 MB.
    """

    def __init__(self, bottleneck: int = 16, **kw):
        super().__init__(**kw)
        self.bottleneck = bottleneck

    def build(self, input_shape):
        c = int(input_shape[-2])
        self.eff = min(self.bottleneck, c)
        self.project = VNLinear(self.eff, dtype=self.dtype) if self.eff != c else None
        super().build(input_shape)

    @property
    def out_features(self) -> int:
        return self.eff * self.eff

    def call(self, x):
        z = self.project(x) if self.project is not None else x
        # The Gram matrix is a product of activations with themselves, so it
        # overflows in half precision at half the magnitude a mixed product
        # would. Small by design after the bottleneck.
        z = _at_least_float32(z)
        gram = tf.matmul(z, z, transpose_b=True)
        return tf.cast(tf.reshape(gram, tf.concat([tf.shape(z)[:-2], [self.eff * self.eff]], 0)),
                       x.dtype)


# ---------------------------------------------------------------------------
# Rotation head
# ---------------------------------------------------------------------------
def gram_schmidt_frame(a1, a2, eps: float = 1e-8):
    """
    6D rotation representation (Zhou et al., 2019): orthonormalise two
    equivariant 3-vector channels into a proper rotation.

        b1 = normalize(a1)
        b2 = normalize(a2 - <b1,a2> b1)
        b3 = b1 x b2
        F  = [b1 b2 b3]   (as COLUMNS)

    Always a valid rotation (det = +1, guaranteed by the cross product rather
    than an arbitrary third vector). Replaces quaternion regression, which is
    not a linear representation of SO(3) and cannot be produced equivariantly by
    a VN backbone.

    Returns the FRAME, which is LEFT-equivariant: F(Ax) = A F(x). Read
    `predict_rotation` before using it as a predicted rotation.
    """
    a1, a2 = _at_least_float32(a1), _at_least_float32(a2)

    # PRE-NORMALISE both channels to unit length.
    #
    # Gram-Schmidt depends only on the DIRECTIONS of a1 and a2, so this is
    # mathematically identical -- but it makes the routine SCALE-INVARIANT,
    # which the raw form is not. With a fixed eps^2 = 1e-16, the guard is
    # negligible against ||a2_orth||^2 ~ 1 yet comparable to it once the
    # predicted vectors shrink: measured, the frame drifts 2e-9 from orthogonal
    # at ||a|| ~ 1e-3 and 2e-3 at ||a|| ~ 1e-6. The worst pairs are those whose
    # two channels are nearly parallel -- exactly the high-`head_cos` regime
    # real training runs sit in. After normalising, eps only matters for a
    # genuinely degenerate input, where no valid frame exists anyway.
    a1 = a1 / tf.sqrt(tf.reduce_sum(a1 * a1, -1, keepdims=True) + eps * eps)
    a2 = a2 / tf.sqrt(tf.reduce_sum(a2 * a2, -1, keepdims=True) + eps * eps)

    # `sqrt(sum + eps^2)`, not a norm with an added or clamped eps: an additive
    # eps leaves ||b1|| short of 1 by eps/||a1||, and clamping leaves the
    # gradient of `norm` undefined at the zero vector.
    b1 = a1 / tf.sqrt(tf.reduce_sum(a1 * a1, -1, keepdims=True) + eps * eps)
    a2_orth = a2 - tf.reduce_sum(b1 * a2, -1, keepdims=True) * b1
    b2 = a2_orth / tf.sqrt(tf.reduce_sum(a2_orth * a2_orth, -1, keepdims=True) + eps * eps)
    b3 = tf.linalg.cross(b1, b2)
    return tf.stack([b1, b2, b3], axis=-1)


def predict_rotation(a1, a2, eps: float = 1e-8):
    """
    The predicted rotation R_pred = F^T, where F is the Gram-Schmidt frame.

    ### Why the transpose is REQUIRED, not cosmetic

    Every VN primitive is LEFT-equivariant: rotate the input geometry by A and

        F(A x) = A F(x).                                                  (1)

    The supervision target points the other way. The network is fed the diffused
    fragment (geometry A x, A the unknown scattering rotation) and must output
    the rotation that undoes it:

        R_gt = A^T.                                                       (2)

    If the head returned F directly, (1) and (2) would force
    F(x_clean) = (A^2)^T for the clean geometry. But x_clean does not depend on
    A -- the same fragment is scattered by a different random A on every draw --
    so one F(x_clean) would have to equal (A^2)^T for every A simultaneously. It
    cannot. The target is unlearnable by ANY left-equivariant head, and the
    failure is silent: training plateaus near chance with no error.

    Transposing fixes it. With G(x) := F(x)^T, (1) becomes

        G(A x) = (A F(x))^T = F(x)^T A^T = G(x) A^T,                      (3)

    i.e. G is RIGHT-equivariant. A single, A-independent thing then has to be
    learned -- G(x_clean) = I on canonically-oriented geometry -- and (3)
    delivers R_gt for EVERY A automatically.
    """
    return tf.linalg.matrix_transpose(gram_schmidt_frame(a1, a2, eps=eps))


def geodesic_rotation_loss(R_pred, R_gt):
    """
    Geodesic angle on SO(3) in radians, as atan2(sin, cos) rather than
    arccos((tr - 1)/2), taking

        cos(theta) = (tr(R) - 1) / 2
        sin(theta) = ||[R32-R23, R13-R31, R21-R12]|| / 2

    Two reasons, both practical:

    * ARCCOS NEEDS A CLAMP AND THE CLAMP IS A FLOOR. Guarding the domain means a
      numerically perfect prediction reports sqrt(2*eps) -- about 0.026 degrees
      -- instead of zero.
    * ARCCOS'S GRADIENT DIVERGES WHERE TRAINING ENDS UP. d/dx arccos(x) is
      -1/sqrt(1-x^2), which blows up as the prediction converges, so this term's
      gradient grows without bound the better the model gets and crowds out
      every other loss term. The atan2 form is bounded everywhere.

    The eps inside the sqrt matters too: the axis vector is exactly zero when
    the residual is symmetric (theta = 0 or pi), where a bare norm has an
    undefined gradient that propagates NaN into every weight. It leaves a floor
    of about 3e-5 degrees on a perfect prediction -- four orders of magnitude
    below anything this task measures, and the price of a finite gradient at
    both endpoints.
    """
    R_pred = _at_least_float32(R_pred)
    R_gt = tf.cast(_at_least_float32(R_gt), R_pred.dtype)
    diff = tf.matmul(R_pred, R_gt, transpose_a=True)
    trace = tf.linalg.trace(diff)
    cos_theta = (trace - 1.0) / 2.0
    axis = tf.stack([diff[..., 2, 1] - diff[..., 1, 2],
                     diff[..., 0, 2] - diff[..., 2, 0],
                     diff[..., 1, 0] - diff[..., 0, 1]], axis=-1)
    sin_theta = tf.sqrt(tf.reduce_sum(axis * axis, -1) + 1e-12) / 2.0
    return tf.atan2(sin_theta, cos_theta)
