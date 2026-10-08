"""
Evaluation without Breaking Bad's shared break vertices.

A Breaking Bad object's fragments are cut from one mesh, so the two sides of a
break share the same vertices: a correct match is the same point on both
fragments, a RANSAC fit on correct matches is exact, and the checked placement
lines the pieces up to round-off. Pipelines that sample points on each
fragment independently (GARF and most published tables) never see that. The
two perturbations here take it away, so a number from the matched route can
be read beside theirs:

* **jitter** -- independent Gaussian noise on every input vertex before the
  network sees the scene, ``sigma`` in units of the scene's largest
  fragment's radius (the scale the matching works in), so the two sides of a
  break no longer coincide. Each edge's relative-position feature is
  recomputed to match; normals and every label are left alone.
* **drop** -- this share of the break vertices (the match candidates) left out
  of the matching at random, so many points lose their partner.

They are ``probe_val.py``'s ``--jitter`` and ``--drop``, which measured the
matched rotations this way, now for the whole assembly. ``probe_val.py`` adds
its noise to the normalised coordinates directly; that is the same noise under
the scene normalisation, the default every run so far used, and here it stays
in largest-fragment radii under ``--normalize_mode fragment`` too.

The score is not perturbed: :func:`~reassembly.assembly.scoring.score_batch`
applies the predicted pose to the clean fragment, so the noise changes what the
method sees and not what it is measured against.

Every draw is tied to the scene's name and the seed -- one stream per scene and
per perturbation, as the RANSAC draws are -- so two evaluations of one
checkpoint, with two placements say, see the same noise.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch
from torch import Tensor


def check_noise(jitter: float, drop: float) -> None:
    """A ``ValueError`` naming the setting that cannot be used, else nothing."""
    if not (math.isfinite(jitter) and jitter >= 0.0):
        raise ValueError(f"--jitter must be a finite number >= 0, got {jitter!r}")
    if not 0.0 <= drop < 1.0:
        raise ValueError(f"--drop must be at least 0 and below 1, got {drop!r}: at 1 every "
                         f"break vertex is gone and the matching falls back to whole fragments")


def noise_suffix(jitter: float, drop: float) -> str:
    """``_jitter0.01_drop0.5`` for output file names; empty without noise."""
    return (f"_jitter{jitter:g}" if jitter else "") + (f"_drop{drop:g}" if drop else "")


def scene_generator(key: str, seed: int, device, stream: str = "") -> torch.Generator:
    """
    One random stream per scene, keyed by its name: a scene draws the same
    numbers whatever batch it lands in. ``stream`` names the perturbation;
    the empty name is the stream the RANSAC hypotheses have always used. (A
    digest, not ``hash()``, which Python salts per process.)
    """
    text = f"{key}\x1f{seed}" + (f"\x1f{stream}" if stream else "")
    digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
    generator = torch.Generator(device=device)
    generator.manual_seed(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return generator


def jitter_inputs(batch, sigma: float, seed: int = 0):
    """
    ``batch`` with Gaussian noise of ``sigma`` largest-fragment radii on every
    input vertex, drawn scene by scene. Each edge's relative position
    (``p_source - p_destination``) is recomputed from the noisy vertices;
    normals, labels and targets are untouched. The batch must carry its
    perturbed copy (:func:`~reassembly.data.features.complete_batch`).
    ``sigma == 0`` returns the batch itself.
    """
    if not sigma:
        return batch
    node = batch.node_features.clone()
    fragment = batch.vertex_fragment.long()
    unit = batch.unit.to(device=node.device, dtype=node.dtype)
    fragment_ptr = batch.fragment_ptr.tolist()
    vertex_ptr = batch.vertex_ptr.tolist()
    for scene in range(batch.num_scenes):
        f0, f1 = fragment_ptr[scene], fragment_ptr[scene + 1]
        if f1 == f0:
            continue
        v0, v1 = vertex_ptr[f0], vertex_ptr[f1]
        key = batch.scene_keys[scene] if batch.scene_keys else str(scene)
        generator = scene_generator(key, seed, node.device, "jitter")
        noise = torch.randn((v1 - v0, 3), generator=generator, device=node.device,
                            dtype=node.dtype)
        # sigma largest radii in the world is sigma * max / unit in a fragment's
        # own normalised coordinates: 1 under the scene normalisation.
        scale = sigma * unit[f0:f1].max() / unit[fragment[v0:v1]]
        node[v0:v1, 0, :] = node[v0:v1, 0, :] + noise * scale[:, None]
    edge = batch.edge_attr.clone()
    edge[:, 2, :] = node[batch.edge_index[0], 0, :] - node[batch.edge_index[1], 0, :]
    return batch._replace(node_features=node, edge_attr=edge)


def drop_candidates(candidates: Optional[Tensor], share: float,
                    generator: torch.Generator) -> Optional[Tensor]:
    """``candidates`` with each one kept with probability ``1 - share``;
    unchanged when ``share`` is 0 or there is no mask."""
    if not share or candidates is None:
        return candidates
    keep = torch.rand(candidates.shape, generator=generator,
                      device=candidates.device) >= share
    return candidates & keep
