"""Shared geometry-embedding logic for 3D-extended VLM model adapters.

Hosts the cartesian_fourier 3D position embedding module that turns each
patch's xyz offset (depth times ray direction) plus radial terms and the
unit ray direction into a hidden_size vector added to image_embeds.
Multiple model classes (Qwen3VL3DModel,
Qwen3_5_3DModel) previously each duplicated this code; the duplicate
``Qwen3_5`` copy silently dropped the ``cartesian_fourier`` dispatch
branch, training with zero geometry signal until this mixin landed.

To extend a new VLM family, subclass ``GeometryEmbeddingMixin`` alongside
the upstream HF model class:

    class MyVLM3DModel(GeometryEmbeddingMixin, MyVLMModel):
        def forward(self, ...):
            ...
            image_embeds = self._apply_geometry_embeddings(
                image_embeds,
                projected_depth_bins,
                projected_ray_dirs,
                device,
            )

Inline-patch logic stays on the per-model class — it lives close to the
PATH B inline-patch handling and isn't shared.

Used by Qwen3VL3DModel and Qwen3_5_3DModel. Any new
3D backbone should subclass this mixin rather than re-implementing the
cartesian_fourier position encoder.
"""
import math

import torch


# Module-level latest-stats dict used by DepthMagnitudeCallback. Lives at
# module scope (not on the model) because the model is wrapped in PEFT +
# DeepSpeed at training time; attribute lookups via ``getattr(wrapped, ...)``
# do not reliably descend through every wrapper. Imported by name from
# both ``model_adapters.qwen3_vl.model`` (back-compat re-export) and the
# training callback in ``training/onecanvas/train/train.py``.
DEPTH_VISUAL_STATS: dict = {}


class GeometryEmbeddingMixin:
    """Additive cartesian_fourier 3D position embedding.

    Encodes each patch's xyz offset as per-axis Fourier features plus radial
    and horizontal-radial terms plus the unit ray direction, decodes through
    a 2-layer MLP, scales by a learned gate, and adds the result to the patch
    feature.

    Mixin: provides instance methods only, no ``__init__``. The host class
    must subclass ``torch.nn.Module`` (any HF ``PreTrainedModel`` already
    satisfies this) so submodules registered via ``self.<name> = Linear(...)``
    behave correctly.
    """

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Depth embedding
    # ------------------------------------------------------------------
    def init_depth_embedding(self, hidden_size, mode="cartesian_fourier",
                             num_freqs=16, mlp_hidden=512,
                             cartesian_fourier_use_rmsnorm=False,
                             cartesian_fourier_per_channel_gate=False,
                             cartesian_fourier_gate_init=1.0,
                             cartesian_fourier_mlp_init_std=0.02):
        """Create the depth embedding module. Called from training setup.

        Only ``mode='cartesian_fourier'`` is supported (the shipped encoder):
        each of x, y, z is Fourier-encoded at ``num_freqs`` log-spaced
        frequencies plus the raw value, ray directions are kept as raw channels,
        and a 2-layer MLP + learnable scalar gate decodes the result. Requires
        ray_dirs alongside raw depths at apply time.
        """
        self._depth_embed_mode = mode
        if mode == "cartesian_fourier":
            # Fourier-encoded 3D cartesian coordinates -> 2-layer MLP decoder
            # + learnable scalar gate. Each of x, y, z gets sin/cos at num_freqs
            # log-spaced frequencies plus the raw value, giving 3*(1+2F) channels.
            # Ray directions kept as raw 3 channels. Two radial primitives
            # (‖p‖, r_xz) each get their own Fourier bank at F'=8 frequencies,
            # adding 2*(1+2F') channels. Total input: 3*(1+2F) + 3 + 2*(1+2F') =
            # 136 channels for F=16, F'=8.
            #
            # Low-freq band extended to 0.1 rad/m (wavelength ~63 m) so room-scale
            # quantities (footprint, room size) get unique phase coverage; the
            # previous 1.0 rad/m lower bound aliased anything > ~6 m onto the raw
            # v channel only. The MLP lets the bands be combined nonlinearly into
            # metric primitives (distance, area) before joining the residual —
            # a linear projection alone cannot combine sin/cos across frequencies.
            freqs = torch.exp(torch.linspace(
                math.log(0.1), math.log(100.0), num_freqs))
            self.register_buffer("_depth_fourier_freqs", freqs)
            # Unary magnitude primitives ‖p‖ (scene-center distance) and
            # r_xz = sqrt(x² + z²) (horizontal radial, y-up convention) each
            # get their own smaller room-scale Fourier bank. These are
            # nonlinear in xyz and cannot be produced by a bilinear attention
            # on axis-aligned Fourier channels alone. ‖p_A‖² + ‖p_B‖² − 2 p_A·p_B
            # becomes a single-layer read for pairwise distance, and r_xz
            # directly supplies room-footprint content per token.
            norm_freqs = torch.exp(torch.linspace(
                math.log(0.1), math.log(10.0), 8))
            self.register_buffer("_depth_norm_fourier_freqs", norm_freqs)
            enc_dim = (3 * (1 + 2 * num_freqs) + 3
                       + 2 * (1 + 2 * norm_freqs.shape[0]))
            self.depth_cartesian_fourier_mlp = torch.nn.Sequential(
                torch.nn.Linear(enc_dim, mlp_hidden, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(mlp_hidden, hidden_size, bias=False),
            )
            # MLP init std. 0.02 is the shipped value and every published
            # number used it. It leaves the depth branch VERY quiet at step 0:
            # measured against the stage-1 feature stash, mean‖depth_emb‖ is
            # 1.07 against a visual norm of 17.19, a ratio of 0.062, and the
            # branch has to climb ~40x before it carries anything. The healthy
            # runs get there by growing these weights to std ~0.060 (3x) while
            # the gate falls to ~0.70. Raising the init lets the branch START
            # near that magnitude instead of travelling to it under Adam, whose
            # per-step size is absolute, so the same absolute travel is a much
            # smaller RELATIVE growth from a larger start.
            #
            # It is NOT interchangeable with raising the gate. Adam's update is
            # invariant to a uniform scaling of the gradients, so a gate of g
            # multiplies every MLP gradient by g and Adam divides it straight
            # back out: measured, two runs at gate 16.77 and 30.19 had
            # pre-gate MLP output 1.428 and 1.440 at step 1000, identical to
            # within 1%. The gate scales the branch's output, the init moves
            # where it starts in parameter space, and only the second changes
            # how fast it grows.
            torch.nn.init.normal_(
                self.depth_cartesian_fourier_mlp[0].weight,
                mean=0.0, std=cartesian_fourier_mlp_init_std)
            torch.nn.init.normal_(
                self.depth_cartesian_fourier_mlp[2].weight,
                mean=0.0, std=cartesian_fourier_mlp_init_std)
            # Optional RMSNorm on the MLP output. Decouples output magnitude
            # from MLP weight growth: post-norm ‖.‖ ≈ √hidden_size regardless
            # of how large the MLP weights become, so amplitude is controlled
            # by the gate scalar only. Pair with a small gate_init (e.g. 0.025)
            # to land depth_emb at ~10% of visual norm.
            if cartesian_fourier_use_rmsnorm:
                self.depth_cartesian_fourier_norm = torch.nn.RMSNorm(
                    hidden_size, eps=1e-6)
            # Gate: scalar (default) or per-channel [hidden_size]. Per-channel
            # is a row-rescaling of the second linear -- same hypothesis class
            # as the MLP, but separate Adam state per channel.
            if cartesian_fourier_per_channel_gate:
                gate_shape = (hidden_size,)
            else:
                gate_shape = (1,)
            self.depth_cartesian_fourier_gate = torch.nn.Parameter(
                torch.full(gate_shape, float(cartesian_fourier_gate_init)))
        else:
            raise ValueError(
                f"unsupported depth_embed_mode {mode!r}; only "
                "'cartesian_fourier' is supported")

    def init_depth_embed_learned_scale(self):
        """Add a learnable exp(param) scalar on depth embeddings.

        Initialized at 0.0 so exp(0)=1.0 (neutral start). The model learns
        how loud the depth signal should be. Unbounded above 1 (amplify) and
        smoothly decays toward 0 (attenuate). Read the converged value via
        exp(model.depth_embed_log_scale.item()).
        """
        self.depth_embed_log_scale = torch.nn.Parameter(torch.tensor(0.0))

    def _encode_depth_cartesian_fourier(self, raw_depths, ray_dirs):
        """Thin wrapper: patches' xyz is depth × ray_dir, direction = ray_dir."""
        xyz = raw_depths.unsqueeze(-1) * ray_dirs   # [N, 3] in meters
        return self._encode_cartesian_fourier(xyz, ray_dirs)

    def _encode_cartesian_fourier(self, xyz, direction):
        """Fourier-encoded 3D cartesian coordinates -> 2-layer MLP + gate.

        Each of x, y, z is expanded with sin/cos at multiple frequencies (NeRF
        positional encoding). The 3-channel ``direction`` vector passes through
        as raw channels. Two unary magnitude primitives — ‖p‖ (scene-center
        distance) and r_xz = sqrt(x² + z²) (horizontal radial, y-up) — each
        get their own smaller room-scale Fourier bank. The full vector is
        decoded by a 2-layer MLP (Linear -> GELU -> Linear) and scaled by a
        learnable scalar gate.

        Called by ``_encode_depth_cartesian_fourier`` for canvas patches
        (direction = unit ray from pano-center through patch) and by the
        ``cartesian_fourier`` camera marker path (xyz = camera center,
        direction = unit pano-center-to-camera ray — making a camera marker
        occupy the same feature subspace as a hypothetical patch at the
        camera body's canvas location).

        With F=16 axis frequencies and F'=8 norm frequencies:
          per axis: [v, sin/cos × F]             → 3 × 33 = 99
          direction:                             →        3
          norm primitives (‖p‖, r_xz):           → 2 × 17 = 34
          total                                  →       136

        Args:
            xyz:       [N, 3] scene-centered positions (meters).
            direction: [N, 3] 3-channel direction vector (unit for patches/
                       cameras; any 3 raw channels for other uses).

        Returns:
            [N, hidden_size] fourier-encoded cartesian position embeddings.
        """
        freqs = self._depth_fourier_freqs.to(xyz.device)  # [F]
        # phases: [N, 3, F]
        phases = xyz.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        sin_feat = torch.sin(phases)    # [N, 3, F]
        cos_feat = torch.cos(phases)    # [N, 3, F]
        raw_xyz = xyz.unsqueeze(-1)     # [N, 3, 1]
        # Per-axis: [raw, sin_f1..sin_fF, cos_f1..cos_fF] → [N, 3, 1+2F]
        per_axis = torch.cat([raw_xyz, sin_feat, cos_feat], dim=-1)
        # Explicit channel count so reshape is unambiguous when N=0.
        n_per_axis = 1 + 2 * freqs.shape[0]
        per_axis = per_axis.reshape(xyz.shape[0], 3 * n_per_axis)

        # Unary norms: ‖p‖ and r_xz (y-up horizontal radial).
        p_norm = xyz.norm(dim=-1, keepdim=True)                      # [N, 1]
        r_xz = (xyz[:, 0:1] ** 2 + xyz[:, 2:3] ** 2).sqrt()          # [N, 1]
        norms = torch.cat([p_norm, r_xz], dim=-1)                    # [N, 2]
        norm_freqs = self._depth_norm_fourier_freqs.to(xyz.device)   # [F']
        norm_phases = norms.unsqueeze(-1) * norm_freqs.unsqueeze(0).unsqueeze(0)
        norm_sin = torch.sin(norm_phases)                            # [N, 2, F']
        norm_cos = torch.cos(norm_phases)                            # [N, 2, F']
        norm_raw = norms.unsqueeze(-1)                               # [N, 2, 1]
        norm_per = torch.cat([norm_raw, norm_sin, norm_cos], dim=-1)  # [N, 2, 1+2F']
        n_per_norm = 1 + 2 * norm_freqs.shape[0]
        norm_per = norm_per.reshape(xyz.shape[0], 2 * n_per_norm)

        encoded = torch.cat([per_axis, direction, norm_per], dim=-1)
        encoded = encoded.to(self.depth_cartesian_fourier_mlp[0].weight.dtype)
        out = self.depth_cartesian_fourier_mlp(encoded)
        if hasattr(self, "depth_cartesian_fourier_norm"):
            out = self.depth_cartesian_fourier_norm(out)
        return self.depth_cartesian_fourier_gate * out

    def _depth_ratio_penalty_value(self, ratio):
        """Aggregate depth/visual norm ratio (a scalar) -> scalar penalty, per
        the configured mode. ``ratio`` is mean‖depth‖ / mean‖visual‖, i.e. the
        same quantity logged as ``depth/visual_ratio``. Shared by the loss path
        (``_add_depth_with_penalty``) and the logging path
        (``_stash_depth_visual_stats``) so the two never diverge. Returns
        ``None`` when disabled (beta <= 0).

        Operating on the AGGREGATE ratio (not a per-token mean of d_i/v_i) is
        deliberate. A single token with near-zero visual norm makes its own
        d_i/v_i blow up to ~1e8, and the per-token mean of squares then explodes
        the loss (observed: penalty ~5e14 with the old per-token form). The
        ratio of means has a denominator of ~tens that no single degenerate
        token can collapse, so it stays bounded and exactly tracks the metric.

        - ``"hinge"`` (default): ``beta * relu(ratio - r0)^2`` — a soft budget.
          Free below ``r0`` (zero gradient there, so depth can neither be driven
          below the budget nor collapsed by this term); every unit of ratio
          above ``r0`` costs quadratically, a non-saturating restoring force
          that grows with the excess so a too-loud depth is pulled back toward
          ``r0`` rather than parked, while useful depth can still exceed it.
        - ``"l2"``: legacy ``beta * ratio^2`` (minimum at zero), prefer hinge.
        """
        beta = float(getattr(self, "_depth_ratio_penalty_beta", 0.0))
        if beta <= 0:
            return None
        mode = getattr(self, "_depth_ratio_penalty_mode", "hinge")
        if mode == "l2":
            return beta * ratio.pow(2)
        r0 = float(getattr(self, "_depth_ratio_penalty_r0", 0.5))
        return beta * (ratio - r0).clamp_min(0.0).pow(2)

    def _stash_depth_visual_stats(self, image_embeds, depth_emb):
        with torch.no_grad():
            d = depth_emb.detach().float()
            v = image_embeds.detach().float()
            d_tok = d.norm(dim=-1)
            v_tok = v.norm(dim=-1)
            d_norm = d_tok.mean().item()
            v_norm = v_tok.mean().item()
            ratio = d_norm / max(v_norm, 1e-8)
            gate = getattr(self, "depth_cartesian_fourier_gate", None)
            stats = {
                "depth/per_token_norm": d_norm,
                "depth/visual_ratio": ratio,
            }
            if gate is not None:
                stats["depth/gate"] = gate.detach().float().mean().item()
            # Penalty diagnostics: the penalty value (so beta can be tuned from
            # the dashboard) and, for the hinge, the fraction of tokens whose
            # own depth norm exceeds r0 x the *mean* visual norm (a robust
            # per-token view that a near-zero-norm token cannot skew).
            beta = float(getattr(self, "_depth_ratio_penalty_beta", 0.0))
            if beta > 0:
                pen = self._depth_ratio_penalty_value(torch.as_tensor(ratio))
                if pen is not None:
                    stats["depth/penalty"] = float(pen)
                if getattr(self, "_depth_ratio_penalty_mode", "hinge") != "l2":
                    r0 = float(getattr(self, "_depth_ratio_penalty_r0", 0.5))
                    v_ref = max(v_norm, 1e-6)
                    stats["depth/frac_over_budget"] = (d_tok / v_ref > r0).float().mean().item()
            DEPTH_VISUAL_STATS["last"] = stats

    def _add_depth_with_penalty(self, image_embeds, depth_emb):
        """Add ``depth_emb`` to ``image_embeds`` and optionally compute the
        depth/visual ratio penalty on the AGGREGATE ratio mean‖depth‖ /
        mean‖visual‖ (see ``_depth_ratio_penalty_value`` for the shape and why
        aggregate rather than per-token). The penalty is stashed on
        ``self._last_depth_ratio_penalty`` for the Trainer's ``compute_loss`` to
        pick up. The visual norm is detached so the penalty cannot be reduced by
        inflating visual; only depth can move. Active only while training.
        """
        beta = float(getattr(self, "_depth_ratio_penalty_beta", 0.0))
        if beta > 0 and self.training:
            d_mean = depth_emb.norm(dim=-1).mean()
            v_mean = image_embeds.norm(dim=-1).mean().detach().clamp_min(1e-6)
            self._last_depth_ratio_penalty = self._depth_ratio_penalty_value(d_mean / v_mean)
        else:
            self._last_depth_ratio_penalty = None
        return image_embeds + depth_emb

    # ------------------------------------------------------------------
    # Top-level dispatch: call from forward() to apply the position embedding
    # ------------------------------------------------------------------
    def _apply_geometry_embeddings(
        self,
        image_embeds,
        projected_depth_bins,
        projected_ray_dirs,
        device,
    ):
        """Apply the depth additive embedding to image_embeds.

        Dispatch on ``self._depth_embed_mode``; each branch is a no-op when
        the corresponding embedding module wasn't built (e.g. when
        ``init_depth_embedding`` was never called or the dataloader didn't
        produce the required inputs). Returns the updated image_embeds.
        """
        # --- Add depth embedding if enabled ---
        _de_mode = getattr(self, "_depth_embed_mode", "off")
        _de_scale = float(getattr(self, "_probe_depth_scale", 1.0))
        _learned_log_scale = getattr(self, "depth_embed_log_scale", None)
        if _de_mode == "cartesian_fourier":
            # EVERY way of not applying the embedding warns, once. The old
            # shape warned on exactly one of the four (bins/rays handed in as
            # None) and stayed silent on the three that actually happen, which
            # is how three eval harnesses ran for weeks with the metric
            # channel amputated (agentic-onecanvas docs/NOW.md 2026-08-01b).
            # The common case is bins arriving as a LIST OF NONES rather than
            # None, which passes an `is not None` guard and then yields an
            # empty valid list.
            why = None
            if not hasattr(self, "depth_cartesian_fourier_mlp"):
                why = ("the model has no depth_cartesian_fourier_mlp, so "
                       "configure_model_3d / init_depth_embedding never ran")
            elif projected_depth_bins is None or projected_ray_dirs is None:
                why = "projected_depth_bins or projected_ray_dirs is None"
            else:
                valid_depths = [b.to(device) for b in projected_depth_bins if b is not None]
                valid_rays = [r.to(device) for r in projected_ray_dirs if r is not None]
                if not valid_depths or not valid_rays:
                    why = (f"every entry is None "
                           f"({len(projected_depth_bins)} bins, "
                           f"{len(projected_ray_dirs)} ray dirs), which means "
                           f"the dataloader ran with use_depth_embedding off")
                elif len(valid_depths) != len(valid_rays):
                    why = (f"{len(valid_depths)} depth bins against "
                           f"{len(valid_rays)} ray dirs")
                else:
                    all_depths = torch.cat(valid_depths, dim=0).float()
                    all_rays = torch.cat(valid_rays, dim=0).float()
                    depth_emb = self._encode_depth_cartesian_fourier(
                        all_depths, all_rays).to(image_embeds.dtype)
                    _fixed = float(getattr(self, "_depth_fixed_ratio", 0.0))
                    if _fixed > 0:
                        # Pin mean||depth|| / mean||visual|| at _fixed by
                        # construction (see DataArguments.depth_embed_fixed_ratio).
                        # Visual side detached, depth side in the graph: the
                        # branch can only move direction, and the scalar gate
                        # cancels exactly. Aggregate over tokens, not per token,
                        # so relative per-token loudness survives and one
                        # degenerate token cannot blow the factor up.
                        _v = image_embeds.detach().float().norm(dim=-1).mean()
                        _d = depth_emb.float().norm(dim=-1).mean().clamp_min(1e-6)
                        depth_emb = (depth_emb.float() * (_fixed * _v / _d)
                                     ).to(image_embeds.dtype)
                    if _de_scale != 1.0:
                        depth_emb = depth_emb * _de_scale
                    if _learned_log_scale is not None:
                        depth_emb = depth_emb * torch.exp(_learned_log_scale)
                    self._stash_depth_visual_stats(image_embeds, depth_emb)
                    image_embeds = self._add_depth_with_penalty(image_embeds, depth_emb)
            if why is not None and not getattr(
                    self, "_warned_cart_fourier_no_rays", False):
                print("[depth_embed][warn] depth_embed_mode=cartesian_fourier "
                      f"but the 3D position embedding is NOT being applied: "
                      f"{why}. The model keeps angular MRoPE, so it will read "
                      "direction and not distance, and any metric number "
                      "measured like this is void.")
                self._warned_cart_fourier_no_rays = True
        return image_embeds
