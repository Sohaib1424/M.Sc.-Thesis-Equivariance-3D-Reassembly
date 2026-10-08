# Review of the GAT / VN / V-GAT figures

Three explanatory posters were produced for the thesis: a GAT walkthrough, a
Vector Neuron layer, and a combined "V-GAT". They are useful as exposition, but
**the V-GAT figure specifies a layer that is not rotation-equivariant**, despite
a Properties box claiming it is. If anyone implements from that poster they will
get a model that trains, descends, and cannot reach the target.

Everything below was checked by running the figures' own formulas, not by
reading them. The numbers are reproducible.

---

## 1. V-GAT — the attention score is not invariant

The figure gives

```
e_ij = LeakyReLU( a₁ᵀ(z_i ⊙ z_j) + a₂ᵀ r_ij + a₃ᵀ(z_i ⊙ r_ij) + a₄ᵀ z_f_ik )
```

Two things are wrong with it, and the second is fatal.

**It is not a scalar.** `z_i ⊙ z_j` is an element-wise product of two `(d_h, 3)`
tensors, so it is still `(d_h, 3)`. Contracting with `a₁ ∈ ℝ^{d_h}` consumes the
*channel* axis and leaves the *spatial* one — the result is a 3-vector. An
attention logit has to be one number.

**It is not invariant**, so `α_ij` changes when the fragment is re-posed.
Measured on random inputs under one random rotation:

| term | change under rotation |
|---|---|
| `a₁ᵀ(z_i ⊙ z_j)` | 1.96 |
| `a₂ᵀ r_ij` | 0.50 |
| `a₃ᵀ(z_i ⊙ r_ij)` | 5.42 |
| `a₄ᵀ z_f_ik` | 0.97 |
| **whole score** | **5.57** |

Invariant would be 0. `a₂ᵀ r_ij` is the clearest case: `a₂` is a *learned fixed
direction* dotted with a rotating vector. That is the same mistake the handoff
document already records for the virtual nodes' fixed spatial anchors — a
constant vector does not rotate when the input does.

**The fix** is to contract the *spatial* axis rather than the channel axis, so
every term is an inner product of two things that rotate together:
`⟨z_i, z_j⟩`, `⟨z_i, r_ij⟩`, `‖z_i‖`. `nn/gat.py` does exactly this — it projects
to `heads × head_dim` directions with a `VNLinear` and sums `q · k` over both
the head and spatial axes.

## 2. V-GAT — the message function is not equivariant either

```
m_ij = σ( U₁ z_j + U₂ (r_ij ⊙ z_j) + U₃ z_f_ik )
```

| variant | equivariance error |
|---|---|
| as drawn, `σ` = SiLU | 8.54 |
| `σ` = identity | 6.17 |
| `U₂(r ⊙ z_j)` alone | 6.17 |
| `U₂ ⟨z_j, r̂⟩ z_j` (the fix) | **0.00** |

Two separate breaks:

- **`σ` = SiLU.** A pointwise nonlinearity acts per *coordinate*, which is a
  statement about the axes of the coordinate system. It must be a VN
  nonlinearity — reflect about a learned, equivariant hyperplane
  (`VNLeakyReLU`), not clamp x, y and z independently.
- **`r_ij ⊙ z_j`.** Element-wise multiplication by a direction's components
  scales each axis differently — a shear, not a rotation-commuting map. The
  equivariant way to combine a direction with a vector feature is a *scalar*
  coefficient times a vector: `⟨z_j, r̂⟩ r̂`, `⟨z_j, r̂⟩ z_j`, or a cross product.

`U₁ z_j` and `U₃ z_f_ik` are fine — a matrix on the channel axis commutes with
a rotation on the spatial axis. That is the one part of the message that is
already a genuine Vector Neuron operation.

Minor: the figure writes `W_in ∈ ℝ^{d_in × d_h}` and then uses it as
`z = W_in n`, which needs `ℝ^{d_h × d_in}`. Transposed.

## 3. VN figure — it does not describe a Vector Neuron layer

What the poster calls "the VN layer" is

```
r̂_ij = (p_j − p_i)/‖p_j − p_i‖
a_ij = (x_j · r̂_ij) r̂_ij          directional projection
m_ij = φ(r_ij) a_ij                radial scaling
m̄_i  = Σ_j m_ij                    aggregate
z_i  = W m̄_i                       linear map on channels
```

That is an equivariant *message-passing* layer in the EGNN / TFN ℓ=1 family. A
Vector Neuron layer, as Deng et al. define it, is the set of primitives —
`VNLinear` (bias-free channel mix), `VN-ReLU` (learned half-space),
`VN-BatchNorm`, `VN-Invariant`. Only the last line above is one of them. The
distinction matters because someone reading the poster would conclude that VN
*requires* a neighbourhood, which it does not: `VNLinear` is a pointwise
operation on a single node.

**The directional projection also throws away most of the signal.**
`a_ij = (x_j · r̂) r̂` keeps only the component of the neighbour's feature along
the edge and discards both perpendicular components. Averaged over random
inputs, it retains **33.4%** of the feature's energy — exactly the 1-of-3
dimensions you would expect. For a mesh vertex that means the tangential part
of a neighbour's normal is irrecoverably lost. A VN layer has no such loss:
`VNLinear` is invertible when square.

Also: the summary strip labels the output "**variant** to rotations", which
contradicts the rest of the poster. It should read *equivariant*. And
`VN Layer(R·X) = R·VN Layer(X)` puts `R` on the left of an `(d_in, 3)` tensor,
where it needs to act on the right as `X Rᵀ`.

## 4. GAT figure — structure right, arithmetic decorative

The formulas are textbook Veličković and are correct: `h' = W h`,
`e_ij = LeakyReLU(aᵀ[h'_i ‖ h'_j])`, softmax, weighted sum, multi-head
concat-or-average. Including self-loops in `N(i)` is right too.

The worked example does not survive recomputation:

- `W H` with the figure's own `W` and `H` gives `h'₁ = [0.07, 0.09, 0.13]`; the
  figure prints `[0.05, 0.11, 0.08]`.
- Using the figure's own `H'` and `a`, the scores come out
  `[0.007, −0.005, −0.002, 0.001, −0.006]`; the figure prints
  `[0.026, 0.061, 0.158, 0.026, −0.027]`.
- The printed `α` sum to **0.900**, under a line stating `Σα = 1.000`.
- **Softmax is monotone**, so the largest score must give the largest weight.
  The figure's largest score is node 3 (`e = 0.158`) but its largest weight is
  node 2 (`α = 0.207`). The true softmax of the printed scores is
  `[0.195, 0.202, 0.223, 0.195, 0.185]`.

Nothing conceptual is wrong here; the numbers are illustrative rather than
computed. Worth fixing before anything goes in a thesis, since a reader who
checks will find it.

---

## 5. What the figures had that this implementation was missing

One real gap, now closed. All three posters carry the **relative direction**
`r_ij` as an edge quantity; `data/features.py` carried only the two adjacent
face normals.

It matters because a message in `VNGraphAttention` is a linear map of the
*source* node's features and the edge's. Without a relative-position channel
the message can describe what the neighbour looks like but not **which way it
lies**. The destination's own coordinate enters through a separate self-loop
term, so a difference is expressible in the first layer if the weights conspire
— but after one layer the channels are mixed abstractions and relative position
stops being cleanly recoverable.

Edges now carry `(n₁, n₂, p_src − p_dst)`, three channels. The raw difference
rather than the unit direction: equally equivariant, and the length stays
readable as an invariant instead of being normalised away.

Two things from the figures deliberately **not** adopted:

- **`φ(r_ij)`, a learnable radial function** (VN poster). The attention weights
  already provide a learned per-edge scalar modulation, and they are adaptive
  rather than a fixed function of distance. Worth an ablation, not a default.
- **A pointwise `σ` after the message** (V-GAT poster). Not adoptable at all —
  see §2.

## 6. Reproducing this review

```bash
python -m pytest tests/test_nn_equivariance.py -q     # what correct looks like
```

The equivariance errors quoted in §1 and §2 come from implementing the figures'
formulas literally in float64 and applying one random `R ∈ SO(3)`; the 33.4%
in §3 is a 20,000-sample average of `‖(x·r̂)r̂‖² / ‖x‖²`; the §4 numbers are
direct recomputations of the figure's own matrices.
