"""Standalone EoS-FM inference, following the Pangaea clean encoder protocol."""

from __future__ import annotations

import re
import warnings
from logging import Logger, getLogger
from pathlib import Path
from typing import Callable

import timm
import torch
import torch.nn as nn

__all__ = ["EosFM", "SmartBandAdapter", "EncoderSelector", "WNConv2d"]

# =============================================================================
# Spectral constants & helpers
# =============================================================================

# Center wavelength (nm) for known remote sensing bands
BAND_WAVELENGTH_NM: dict[str, float | None] = {
    # Sentinel-2
    "B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705, "B6": 740,
    "B7": 783, "B8": 842, "B8A": 865, "B9": 945, "B10": 1375, "B11": 1610,
    "B12": 2190,
    # Generic optical
    "R": 665, "G": 560, "B": 490, "NIR": 842, "IR": 842,
    "SWIR1": 1610, "SWIR2": 2190,
    # SAR (no optical wavelength)
    "VV": None, "VH": None, "HH": None, "HV": None,
    # Special
    "CLP": None,
}

SAR_BANDS = frozenset({"VV", "VH", "HH", "HV"})

def _canonicalize_band(band: str) -> str:
    """Normalize a band name to its canonical form (B01→B1, RED→R, etc.)."""
    band = band.strip().upper()
    aliases = {"RED": "R", "GREEN": "G", "BLUE": "B", "BNIR": "NIR", "B08A": "B8A"}
    if band in aliases:
        return aliases[band]
    if band.startswith("B") and len(band) == 3 and band[1] == "0" and band[2].isdigit():
        return "B" + band[2]
    return band


_EXPR_RE = re.compile(r"^([A-Za-z0-9]+)\s*([\-+*/])\s*([A-Za-z0-9]+)$")


def _parse_band_expression(band: str) -> tuple[str, str, str] | None:
    """Parse 'VV-VH' → ('VV', '-', 'VH'). Returns None if not an expression."""
    m = _EXPR_RE.match(band.strip())
    return (m.group(1), m.group(2), m.group(3)) if m else None


def _get_band_modality(band_canonical: str) -> str:
    """Return 'sar' or 'optical' for a canonical band name or expression."""
    if band_canonical in SAR_BANDS:
        return "sar"
    expr = _parse_band_expression(band_canonical)
    if expr:
        left = _get_band_modality(_canonicalize_band(expr[0]))
        right = _get_band_modality(_canonicalize_band(expr[2]))
        return "sar" if (left == "sar" or right == "sar") else "optical"
    return "optical"


# =============================================================================
# Smart Band Adaptation
# =============================================================================


class SmartBandAdapter:
    """Spectral-aware band mapping from available input bands to an encoder's
    training bands.

    Instead of brute-force applying all possible channel-count adaptations,
    finds the single best mapping per encoder using spectral domain knowledge:

    1. **Exact name match** (after alias normalization): score 1.0
    2. **Same-wavelength match** (different name, same λ): score 0.95
    3. **Nearest-wavelength same-modality**: score 0.3–0.9
    4. **Cross-modality fallback with cycling**: score 0.05–0.4

    Band expressions (e.g. ``VV-VH``) are parsed and computed on-the-fly.
    Supported operators: ``- + * /``.
    """

    def __init__(
        self,
        available_bands: list[str],
        encoder_training_bands: list[str],
        encoder_name: str = "",
        available_band_scores: list[float | None] | None = None,
    ):
        if not available_bands or not encoder_training_bands:
            raise ValueError("Available and training band lists must be nonempty")
        self.available_bands = available_bands
        self.encoder_training_bands = encoder_training_bands
        self.encoder_name = encoder_name
        if available_band_scores is not None and len(available_band_scores) != len(available_bands):
            raise ValueError("available_band_scores must have the same length as available_bands")
        self._avail_band_scores = available_band_scores

        self._avail_canonical = [_canonicalize_band(b) for b in available_bands]
        self._enc_canonical = [_canonicalize_band(b) for b in encoder_training_bands]

        self.band_specs, self.quality_scores, self.match_descriptions = self._compute_mapping()
        self._overall_quality = (
            sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0
        )
        self._adapt_callable = self._build_adapt_fn()

    @property
    def overall_quality(self) -> float:
        return self._overall_quality

    # ---- Mapping computation ---------------------------------------------------

    def _compute_mapping(self) -> tuple[list, list[float], list[str]]:
        band_specs: list = []
        scores: list[float] = []
        descriptions: list[str] = []
        cross_modal_cursor: list[int] = [0]
        memo: dict[str, tuple] = {}

        for enc_band_canon in self._enc_canonical:
            spec, score, desc = self._find_best_match(enc_band_canon, cross_modal_cursor, memo)
            band_specs.append(spec)
            scores.append(score)
            descriptions.append(desc)
        return band_specs, scores, descriptions

    def _find_best_match(
        self,
        enc_band: str,
        cross_modal_cursor: list[int],
        memo: dict[str, tuple],
    ) -> tuple:
        if enc_band in memo:
            return memo[enc_band]

        # Handle binary band expressions (e.g. VV-VH)
        expr = _parse_band_expression(enc_band)
        if expr is not None:
            result = self._find_best_match_expression(enc_band, expr, cross_modal_cursor, memo)
            memo[enc_band] = result
            return result

        enc_modality = _get_band_modality(enc_band)
        enc_wl = BAND_WAVELENGTH_NM.get(enc_band)

        # 1. Exact canonical name match
        for i, avail_band in enumerate(self._avail_canonical):
            if avail_band == enc_band:
                result = i, 1.0, f"{enc_band}→{self.available_bands[i]} (exact)"
                memo[enc_band] = result
                return result

        # 2. Same wavelength, different name
        if enc_wl is not None:
            for i, avail_band in enumerate(self._avail_canonical):
                avail_wl = BAND_WAVELENGTH_NM.get(avail_band)
                if avail_wl is not None and avail_wl == enc_wl:
                    result = (i, 0.95, f"{enc_band}→{self.available_bands[i]} (same λ={enc_wl}nm)")
                    memo[enc_band] = result
                    return result

        # 3. Nearest wavelength, same modality
        if enc_wl is not None:
            best_idx, best_dist = -1, float("inf")
            for i, avail_band in enumerate(self._avail_canonical):
                if _get_band_modality(avail_band) != enc_modality:
                    continue
                avail_wl = BAND_WAVELENGTH_NM.get(avail_band)
                if avail_wl is not None:
                    dist = abs(avail_wl - enc_wl)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            if best_idx >= 0:
                score = max(0.3, 1.0 - best_dist / 2000.0)
                result = (best_idx, score, f"{enc_band}→{self.available_bands[best_idx]} (nearest λ, Δ={best_dist:.0f}nm)")
                memo[enc_band] = result
                return result

        # 4. Cross-modality fallback with cycling
        if enc_modality == "sar":
            pool = [i for i, b in enumerate(self._avail_canonical) if _get_band_modality(b) == "sar"]
            fallback_score, label = 0.4, "SAR fallback"
        else:
            pool = [i for i, b in enumerate(self._avail_canonical) if _get_band_modality(b) == "optical"]
            fallback_score, label = 0.1, "optical fallback"

        if not pool:
            pool = list(range(len(self._avail_canonical)))
            fallback_score, label = 0.05, "cross-modality fallback"

        pool = self._rank_pool_by_informativeness(pool)
        idx = pool[cross_modal_cursor[0] % len(pool)]
        cross_modal_cursor[0] += 1
        result = (idx, fallback_score, f"{enc_band}→{self.available_bands[idx]} ({label})")
        memo[enc_band] = result
        return result

    def _find_best_match_expression(
        self,
        enc_band: str,
        expr: tuple[str, str, str],
        cross_modal_cursor: list[int],
        memo: dict[str, tuple],
    ) -> tuple:
        left_canon = _canonicalize_band(expr[0])
        right_canon = _canonicalize_band(expr[2])
        left_spec, left_score, left_desc = self._find_best_match(left_canon, cross_modal_cursor, memo)
        right_spec, right_score, right_desc = self._find_best_match(right_canon, cross_modal_cursor, memo)
        return (expr[1], left_spec, right_spec), min(left_score, right_score), f"{enc_band}→({left_desc} {expr[1]} {right_desc})"

    def _rank_pool_by_informativeness(self, pool: list[int]) -> list[int]:
        if self._avail_band_scores is None:
            return pool
        scores = self._avail_band_scores

        def _score(idx: int) -> float:
            s = scores[idx]
            return float(s) if s is not None else float("-inf")

        return sorted(pool, key=_score, reverse=True)

    # ---- Tensor adaptation -----------------------------------------------------

    def _build_adapt_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Pre-compile the band adaptation into a pure-tensor callable.

        No ``isinstance`` checks at call time — fully torch.compile compatible.
        """

        def _make_fn(spec: "int | tuple") -> Callable[[torch.Tensor], torch.Tensor]:
            if isinstance(spec, int):
                def _fn(image: torch.Tensor, _i: int = spec) -> torch.Tensor:
                    return image[:, _i, :, :]
                return _fn

            op, left_spec, right_spec = spec
            left_fn, right_fn = _make_fn(left_spec), _make_fn(right_spec)
            ops = {
                "-": lambda img, lf=left_fn, rf=right_fn: lf(img) - rf(img),
                "+": lambda img, lf=left_fn, rf=right_fn: lf(img) + rf(img),
                "*": lambda img, lf=left_fn, rf=right_fn: lf(img) * rf(img),
                "/": lambda img, lf=left_fn, rf=right_fn: lf(img) / (rf(img).abs() + 1e-6),
            }
            if op not in ops:
                raise ValueError(f"Unknown band expression operator: '{op}'")
            return ops[op]

        # Fast path: all specs are plain integer indices
        if all(isinstance(s, int) for s in self.band_specs):
            indices: list[int] = list(self.band_specs)  # type: ignore[arg-type]
            return lambda image, _idx=indices: image[:, _idx, :, :]

        # General path: tree of closures
        channel_fns = [_make_fn(spec) for spec in self.band_specs]
        return lambda image, _fns=channel_fns: torch.cat([fn(image).unsqueeze(1) for fn in _fns], dim=1)

    def adapt(self, image: torch.Tensor) -> torch.Tensor:
        """Build the adapted input tensor (B, C_enc, H, W) for the encoder."""
        return self._adapt_callable(image)

    def summary(self) -> str:
        lines = [f"  SmartBandAdapter for '{self.encoder_name}' (overall quality={self._overall_quality:.2f}):"]
        for score, desc in zip(self.quality_scores, self.match_descriptions):
            lines.append(f"    [{score:.2f}] {desc}")
        return "\n".join(lines)


# =============================================================================
# Encoder Selector
# =============================================================================


class EncoderSelector(nn.Module):
    """Hard top-k selection of specialist encoders for a given task.

    Learns a weight per specialist; at forward time, only the top-k encoders
    pass their features through — the rest are zeroed.
    """

    def __init__(
        self,
        num_specialists: int,
        k: int,
        scale_features: bool = False,
        only_val: bool = False,
    ):
        super().__init__()
        if not 1 <= k <= num_specialists:
            raise ValueError("max_encoders must be between 1 and the number of specialists")
        self.num_specialists = num_specialists
        self.k = k
        self.scale_features = scale_features
        self.only_val = only_val
        self.specialists_weights = nn.Parameter(torch.ones(num_specialists))

    def forward(
        self, feature_maps: list[list[torch.Tensor]], features_encoders: list[int]
    ) -> list[list[torch.Tensor]]:
        weights = self.specialists_weights
        k = self.num_specialists if (self.training and self.only_val) else self.k
        topk_weights, topk_indices = torch.topk(weights, k)

        if self.scale_features:
            mask = torch.zeros_like(weights)
            mask.scatter_(0, topk_indices, topk_weights)
        else:
            mask = torch.zeros_like(weights)
            mask.scatter_(0, topk_indices, 1.0)

        result = []
        for i, feats in enumerate(feature_maps):
            enc_idx = features_encoders[i]
            w = mask[enc_idx]
            result.append([f * w for f in feats])
        return result

    def get_deactivated_encoder_indices(self, threshold: float = 0.01) -> set[int]:
        with torch.no_grad():
            return {i for i, w in enumerate(self.specialists_weights) if abs(w.item()) < threshold}

    def get_active_encoder_count(self, threshold: float = 0.01) -> int:
        with torch.no_grad():
            return int(torch.sum(torch.abs(self.specialists_weights) > threshold).item())

    def freeze_specialists_weights(self):
        self.specialists_weights.requires_grad = False


# =============================================================================
# Weight-Normalized Conv2D
# =============================================================================


class WNConv2d(nn.Conv2d):
    """Weight-normalized Conv2d: normalizes filters to unit norm per output channel."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute norm of each output channel filter
        # weight shape: (out_channels, in_channels, H, W)
        weight_reshaped = self.weight.view(self.weight.size(0), -1)  # (out_ch, in_ch*H*W)
        weight_norm = torch.norm(weight_reshaped, dim=1, keepdim=True)  # (out_ch, 1)
        weight_norm = weight_norm.view(self.weight.size(0), 1, 1, 1)  # (out_ch, 1, 1, 1)
        # Normalize: divide each filter by its norm
        normalized_weight = self.weight / (weight_norm + 1e-8)
        return self._conv_forward(input, normalized_weight, self.bias)


# =============================================================================
# EoS-FM Encoder
# =============================================================================


class EosFM(nn.Module):
    """Ensemble of Specialists Foundation Model.

    An ensemble of disjointly-trained specialist encoders whose features are
    fused (and optionally selected before fusion to reduce ensemble size).

    Each specialist is a timm ``features_only`` backbone loaded from a shared
    checkpoint.  Input bands are mapped to each specialist's expected channels
    via :class:`SmartBandAdapter` (spectral-aware, one mapping per encoder).

    Args:
        input_bands: Modality → band names, e.g. ``{"optical": ["B2", "B3", "B4"]}``.
        input_size: Spatial size of input images.
        encoder_weights: Path to ``.pth`` checkpoint (list of ``(in_chans, state_dict, config)``).
        device: Target device.
        feature_fusion: ``None`` (concatenate) or ``"conv1x1"`` (reduce channels).
        fuse_to_mult: Channel multiplier for conv1x1 fusion output.
        finetune_encoders: Unfreeze specialist encoders.
        normalize_features: Apply per-encoder BatchNorm before fusion.
        projection_layer: Add Conv2d 1×1 + LeakyReLU per encoder after normalization.
        max_encoders: If set, use :class:`EncoderSelector` to keep only top-k.
        scale_features: Weight features by selection score (vs binary mask).
        encoder_selector_delay_epochs: Skip selection for the first N epochs.
        encoder_selection_only_val: Apply selection only at validation time.
        activated_encoders: Force-enable only these encoders (others zeroed).
        torch_compile: Compile ``forward()`` with ``torch.compile``.
        torch_compile_mode: Compilation mode (``"reduce-overhead"``, ``"default"``, …).
        bands_std: Per-band standard deviations for informativeness ranking.
        regularize_fusion: Enable the optional fusion L1 penalty.
        trusted_checkpoint: Allow pickle when loading trusted legacy OmegaConf
            checkpoints. Leave False for the compatible release artifact.

    The default concatenates specialist maps without random learned fusion.
    ``feature_fusion="conv1x1"`` explicitly creates trainable WNConv2d filters;
    fusion, projection, selection weights and feature BN statistics are absent
    from the specialist-only release checkpoint. BN updates during ``train()``
    even with frozen parameters; call ``eval()`` to freeze running statistics.

    For reproducible FP32 raw frozen extraction use ``feature_fusion=None,
    normalize_features=False, projection_layer=False, max_encoders=None,
    torch_compile=False`` and call ``model.eval()`` under ``torch.no_grad()``.
    All specialists remain in eval mode unless ``finetune_encoders=True``.

    Adapter fallback and band-expression semantics follow Pangaea: expressions
    operate on supplied (already standardized) channels. Missing CLD has no
    wavelength and uses optical fallback, normally the first optical channel
    (or informativeness ranking when bands_std is supplied). Checkpoint band
    metadata is used as stored without reinterpretation.
    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        encoder_weights: str | Path,
        device: str | torch.device = "cpu",
        feature_fusion: str | None = None,
        fuse_to_mult: int = 1,
        finetune_encoders: bool = False,
        normalize_features: bool = True,
        projection_layer: bool = False,
        max_encoders: int | None = None,
        scale_features: bool = False,
        encoder_selector_delay_epochs: int = 0,
        encoder_selection_only_val: bool = False,
        activated_encoders: list[str] | None = None,
        torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        bands_std: dict[str, list[float]] | None = None,
        regularize_fusion: bool = True,
        trusted_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = "EoS-FM"
        self.input_bands = input_bands
        self.input_size = input_size
        self.encoder_weights = str(encoder_weights)
        self.output_layers = [0, 1, 2, 3]
        self.multi_temporal = False
        self.multi_temporal_output = False
        self.pyramid_output = True
        self.trusted_checkpoint = trusted_checkpoint
        unknown_modalities = set(input_bands) - {"optical", "sar"}
        if unknown_modalities or not any(input_bands.values()):
            raise ValueError("input_bands must contain nonempty optical and/or sar bands")
        if input_size < 32:
            raise ValueError("input_size must be at least 32")

        self.device = torch.device(device)
        self.encoders = nn.ModuleList()
        self.encoders_input_bands: list[int] = []
        self.encoders_out_channels: list[list[int]] = []
        self.encoder_names: list[str] = []
        self.encoders_training_bands: list[list[str]] = []
        self.torch_compile_enabled = torch_compile
        self.torch_compile_mode = torch_compile_mode
        self.bands_std = bands_std
        self.activated_encoders = activated_encoders
        self._deactivated_encoder_indices: set[int] = set()
        self._is_forward_compiled = False
        self._cached_feature_shapes: dict[int, list[tuple[int, ...]]] = {}
        self.regularize_fusion = regularize_fusion
        # Precomputed flag: True once current_epoch >= encoder_selector_delay_epochs.
        # Updated in set_current_epoch() so forward() never reads self.current_epoch
        # (which would cause torch.compile to recompile every epoch).
        self._selector_enabled: bool = False

        # Load all specialist encoders from checkpoint
        self.load_encoder_weights(logger=getLogger(__name__))

        # Force-deactivate encoders not in the activated list
        if self.activated_encoders is not None:
            available = set(self.encoder_names)
            requested = set(self.activated_encoders)
            unknown = sorted(requested - available)
            if unknown:
                raise ValueError(
                    f"Unknown encoder names in activated_encoders: {unknown}. "
                    f"Available: {self.encoder_names}"
                )
            self._deactivated_encoder_indices = {
                i for i, name in enumerate(self.encoder_names) if name not in requested
            }

        if not finetune_encoders:
            self.freeze()

        self.in_chans = sum(len(bands) for bands in input_bands.values())
        self.finetune_encoders = finetune_encoders
        self.normalize_features = normalize_features
        self.projection_layer = projection_layer
        self.encoder_selector_delay_epochs = int(encoder_selector_delay_epochs)
        self.current_epoch: int = 0

        # Build smart band adapters
        self._smart_adapters = self._build_smart_adapters()

        # Compute output dimensions
        self.output_dim = self._compute_out_channels()
        self.embed_dim = list(self.output_dim)
        if not self.output_dim or all(d == 0 for d in self.output_dim):
            raise ValueError("No encoders were loaded successfully.")

        # Encoder selector (top-k)
        self.encoder_selector: EncoderSelector | None = None
        self.max_encoders = max_encoders
        if max_encoders is not None:
            self.encoder_selector = EncoderSelector(
                num_specialists=len(self.encoders),
                k=max_encoders,
                scale_features=scale_features,
                only_val=encoder_selection_only_val,
            )

        # Per-encoder BatchNorm
        self.feature_normalizers = self._build_normalizers()

        # Per-encoder projection (Conv 1×1 + LeakyReLU)
        self.feature_projections = self._build_projections()

        # Feature fusion (None → identity, conv1x1 → learned reduction)
        self.feature_fusion_mode = feature_fusion
        self.fuse_to_mult = fuse_to_mult
        self.feature_fusion = self._build_fusion()

        # Pre-fill shape cache (required before torch.compile)
        self._prefill_feature_shape_cache()

        self._log_parameter_counts()
        self.to(self.device)
        self.out_channels = list(self.output_dim)
        self.set_current_epoch(0)

        if self.torch_compile_enabled:
            self._compile_forward()

    # ---- Initialization helpers ------------------------------------------------

    def _build_smart_adapters(self) -> list[SmartBandAdapter]:
        """Create one SmartBandAdapter per encoder based on spectral knowledge."""
        available_band_names: list[str] = []
        available_band_scores: list[float | None] = []

        for modality in ["optical", "sar"]:
            if modality not in self.input_bands:
                continue
            bands = self.input_bands[modality]
            available_band_names.extend(bands)

            stds: list[float] | None = None
            if self.bands_std is not None and modality in self.bands_std:
                stds = self.bands_std[modality]
                if len(stds) != len(bands):
                    warnings.warn(
                        f"bands_std['{modality}'] length mismatch ({len(stds)} vs {len(bands)}). Ignoring."
                    )
                    stds = None

            if stds is None:
                available_band_scores.extend([None] * len(bands))
            else:
                available_band_scores.extend([float(s) ** 2 for s in stds])

        adapters: list[SmartBandAdapter] = []
        for enc_idx in range(len(self.encoders)):
            training_bands = self.encoders_training_bands[enc_idx]
            expected_chans = self.encoders_input_bands[enc_idx]
            if len(training_bands) != expected_chans:
                if len(training_bands) < expected_chans:
                    raise ValueError(f"Encoder {self.encoder_names[enc_idx]} requires {expected_chans} band names")
                warnings.warn(
                    f"Encoder {self.encoder_names[enc_idx]} declares {len(training_bands)} bands "
                    f"for {expected_chans} channels; retaining the historical leading-band policy.",
                    stacklevel=2,
                )
                training_bands = training_bands[:expected_chans]

            adapter = SmartBandAdapter(
                available_bands=available_band_names,
                encoder_training_bands=training_bands,
                encoder_name=self.encoder_names[enc_idx],
                available_band_scores=available_band_scores,
            )
            adapters.append(adapter)
            getLogger(__name__).debug(adapter.summary())

        return adapters

    def _build_normalizers(self) -> nn.ModuleList:
        normalizers = nn.ModuleList()
        for enc_idx in range(len(self.encoders)):
            levels = nn.ModuleList()
            for ch in self.encoders_out_channels[enc_idx]:
                if self.normalize_features:
                    levels.append(nn.BatchNorm2d(num_features=ch, affine=False))
                else:
                    levels.append(nn.Identity())
            normalizers.append(levels)
        return normalizers

    def _build_projections(self) -> nn.ModuleList:
        projections = nn.ModuleList()
        for enc_idx in range(len(self.encoders)):
            levels = nn.ModuleList()
            for ch in self.encoders_out_channels[enc_idx]:
                if self.projection_layer:
                    levels.append(nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=1),
                        nn.LeakyReLU(inplace=True),
                    ))
                else:
                    levels.append(nn.Identity())
            projections.append(levels)
        return projections

    def _build_fusion(self) -> nn.ModuleList:
        if self.feature_fusion_mode is None:
            return nn.ModuleList([nn.Identity()] * len(self.output_dim))

        if self.feature_fusion_mode != "conv1x1":
            raise ValueError(
                f"Invalid feature_fusion: '{self.feature_fusion_mode}'. Must be None or 'conv1x1'."
            )

        base_channels = self.encoders_out_channels[0]
        target_channels = [int(c * self.fuse_to_mult) for c in base_channels]
        fusion = nn.ModuleList()
        for i, in_ch in enumerate(self.output_dim):
            fusion.append(WNConv2d(in_ch, target_channels[i], kernel_size=1))
        self.output_dim = target_channels
        self.embed_dim = target_channels
        return fusion

    def _compute_out_channels(self) -> list[int]:
        """Sum per-level channels across all encoders (each contributes once)."""
        if not self.encoders:
            raise ValueError("No encoders loaded.")
        num_levels = max(len(ch) for ch in self.encoders_out_channels)
        return [
            sum(ch[lvl] for ch in self.encoders_out_channels if lvl < len(ch))
            for lvl in range(num_levels)
        ]

    def _prefill_feature_shape_cache(self) -> None:
        """Pre-populate feature shapes so forward() never mutates state (torch.compile safe)."""
        dummy = torch.zeros(1, self.in_chans, self.input_size, self.input_size, device=self.device)
        with torch.no_grad():
            for i, encoder in enumerate(self.encoders):
                was_training = encoder.training
                encoder.eval()
                adapted = self._smart_adapters[i].adapt(dummy)
                out = encoder(adapted)
                self._cached_feature_shapes[i] = [tuple(f.shape[1:]) for f in out]
                if was_training:
                    encoder.train()

    # ---- Weight loading --------------------------------------------------------

    def load_encoder_weights(self, logger: Logger):
        if not self.encoder_weights:
            raise ValueError("No encoder weights specified.")
        if len(self.encoders) > 0:
            warnings.warn("Encoders already loaded, skipping.")
            return

        # Compatible release files contain tensors and plain Python metadata only.
        # Legacy OmegaConf pickle files require an explicit trust decision.
        checkpoint = torch.load(
            self.encoder_weights, map_location="cpu", weights_only=not self.trusted_checkpoint
        )
        for in_chans, state_dict, config in checkpoint:
            self.encoders_input_bands.append(in_chans)
            encoder: nn.Module = timm.create_model(
                config["backbone"].removeprefix("timm_"),
                in_chans=in_chans,
                features_only=True,
                pretrained=False,
            )
            encoder.load_state_dict(state_dict, strict=True)
            out_channels = self._compute_out_channels_encoder(encoder, in_chans)
            encoder = encoder.to(self.device)
            self.encoders.append(encoder)
            self.encoders_out_channels.append(out_channels)
            self.encoder_names.append(config.get("encoder_name", f"encoder_{len(self.encoders)}"))

            bands_raw = config.get("bands")
            if bands_raw is None:
                raise ValueError(
                    f"Encoder '{self.encoder_names[-1]}' has no 'bands' field in checkpoint. "
                    "This clean encoder requires checkpoints with explicit band metadata."
                )
            self.encoders_training_bands.append([str(b) for b in bands_raw])
            logger.info("Loaded encoder %s (%d bands)", self.encoder_names[-1], in_chans)
        logger.info("Loaded %d specialist encoders", len(self.encoders))

    @staticmethod
    def _compute_out_channels_encoder(encoder: nn.Module, in_chans: int) -> list[int]:
        if hasattr(encoder, "feature_info"):
            return list(encoder.feature_info.channels())
        was_training = encoder.training
        encoder.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, in_chans, 32, 32)
                return [o.shape[1] for o in encoder(dummy)]
        finally:
            encoder.train(was_training)

    # ---- Freeze / train --------------------------------------------------------

    def freeze(self):
        """Freeze specialist parameters; feature BN still follows train()/eval()."""
        for enc in self.encoders:
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()

    def train(self, mode: bool = True) -> "EosFM":
        super().train(mode)
        if not self.finetune_encoders:
            for enc in self.encoders:
                enc.eval()
        return self

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        # Precompute selector flag outside forward() to avoid per-epoch recompilation.
        # torch.compile will recompile at most once: when this transitions False→True.
        self._selector_enabled = self.current_epoch >= self.encoder_selector_delay_epochs

    # ---- torch.compile ---------------------------------------------------------

    def _compile_forward(self):
        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile not available (requires PyTorch 2.0+), skipping")
            return
        if self._is_forward_compiled:
            return

        print(f"Compiling forward() with mode='{self.torch_compile_mode}'...")
        try:
            _compiled = torch.compile(self.forward, mode=self.torch_compile_mode)

            def _forward_clone(x: dict, _fn=_compiled) -> list[torch.Tensor]:
                return [f.clone(memory_format=torch.contiguous_format) for f in _fn(x)]

            self._original_forward = self.forward
            self.forward = _forward_clone  # type: ignore[assignment]
            self._is_forward_compiled = True
            print("Successfully compiled forward()")
        except Exception as e:
            warnings.warn(f"Failed to compile forward(): {e}")

    # ---- Forward pass ----------------------------------------------------------

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Return feature maps for a tensor or a modality dictionary.

        Tensor channels are optical first, SAR second, in ``input_bands`` order.
        Dictionary values have shape B,C,H,W or B,C,1,H,W. Inputs must already
        have the downstream dataset's normalization; no checkpoint-specific
        input standardization is applied here.
        """
        if isinstance(x, torch.Tensor):
            image = x
        else:
            parts = []
            for modality in ("optical", "sar"):
                if modality not in self.input_bands:
                    continue
                part = x[modality]
                if part.ndim == 5 and part.shape[2] == 1:
                    part = part.squeeze(2)
                if part.ndim != 4 or part.shape[1] != len(self.input_bands[modality]):
                    raise ValueError(f"Invalid {modality} input shape: {tuple(part.shape)}")
                parts.append(part)
            image = torch.cat(parts, dim=1)
        if image.ndim != 4 or image.shape[1] != self.in_chans:
            raise ValueError(f"Expected B,{self.in_chans},H,W input; got {tuple(image.shape)}")
        if self._deactivated_encoder_indices and tuple(image.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError("Static encoder deactivation requires the configured input_size")

        # Statically deactivated encoders (set once at init via activated_encoders).
        # This is a fixed Python set → Dynamo treats it as a constant → no recompilation.
        static_skip = self._deactivated_encoder_indices

        # Run each specialist encoder.
        # Selector-based dynamic deactivation is NOT handled here (would cause per-epoch
        # recompilation). The encoder_selector.forward() applies learned weights (including
        # zeroing non-top-k encoders) as tensor ops, which are compile-stable.
        features: list[list[torch.Tensor]] = []
        encoder_indices: list[int] = []
        for i, encoder in enumerate(self.encoders):
            if i in static_skip and i in self._cached_feature_shapes:
                # Skip forward pass for permanently deactivated encoders.
                B = image.shape[0]
                feats = [image.new_zeros(B, *s) for s in self._cached_feature_shapes[i]]
            else:
                adapted = self._smart_adapters[i].adapt(image)
                feats = encoder(adapted)
                if i in static_skip:
                    feats = [torch.zeros_like(f) for f in feats]

            features.append(feats)
            encoder_indices.append(i)

        # Normalize & project (skip for statically deactivated to avoid BN stat corruption).
        for idx, feats in enumerate(features):
            enc_i = encoder_indices[idx]
            if enc_i in static_skip:
                continue
            features[idx] = [
                self.feature_projections[enc_i][lvl](
                    self.feature_normalizers[enc_i][lvl](feats[lvl])
                )
                for lvl in range(len(feats))
            ]

        # Apply encoder selection.
        # _selector_enabled is precomputed in set_current_epoch() → stable guard,
        # recompiles at most once (False→True transition).
        if self.encoder_selector is not None and self._selector_enabled:
            features = self.encoder_selector(features, encoder_indices)

        # Fuse: concatenate across encoders per level, then apply fusion layer
        num_levels = len(features[0])
        fused = [
            torch.cat([features[e][lvl] for e in range(len(features))], dim=1)
            for lvl in range(num_levels)
        ]
        return [self.feature_fusion[lvl](fused[lvl]) for lvl in range(num_levels)]

    # ---- Export / regularization -----------------------------------------------

    def export_pruned_model(self, selected_encoder_indices: list[int] | None = None) -> "EosFM":
        """Return a compact, independent copy of selected specialists.

        Pruning learned fusion is rejected because changing the input filter norm
        changes WNConv2d's function. Save the full downstream state for that case.
        Raw/BN/projection checkpoints preserve their current trained parameters.
        """
        import copy

        if self.feature_fusion_mode is not None:
            raise ValueError("Pruning learned fusion is unsupported; retain the full model instead")
        if self._is_forward_compiled:
            raise ValueError("Prune before enabling torch_compile")
        if selected_encoder_indices is None:
            if self.encoder_selector is None:
                raise ValueError("Provide selected_encoder_indices or enable encoder selection")
            selected_encoder_indices = sorted(
                torch.topk(self.encoder_selector.specialists_weights, self.max_encoders).indices.tolist()
            )
        indices = list(selected_encoder_indices)
        if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.encoders) for i in indices):
            raise ValueError("Selected encoder indices must be nonempty, unique, and in range")
        # Copy each selected module only, without reopening a temporary checkpoint
        # or losing specialist finetuning / BN / projection state.
        pruned = copy.copy(self)
        pruned.__dict__ = self.__dict__.copy()
        pruned._modules = self._modules.copy()
        for name in ("encoders", "feature_normalizers", "feature_projections"):
            setattr(pruned, name, nn.ModuleList([copy.deepcopy(getattr(self, name)[i]) for i in indices]))
        for name in ("encoder_names", "encoders_input_bands", "encoders_out_channels", "encoders_training_bands", "_smart_adapters"):
            setattr(pruned, name, copy.deepcopy([getattr(self, name)[i] for i in indices]))
        pruned._cached_feature_shapes = {j: self._cached_feature_shapes[i] for j, i in enumerate(indices)}
        pruned._deactivated_encoder_indices = {j for j, i in enumerate(indices) if i in self._deactivated_encoder_indices}
        pruned.encoder_selector = None
        pruned.max_encoders = None
        pruned.activated_encoders = [pruned.encoder_names[j] for j in range(len(indices)) if j not in pruned._deactivated_encoder_indices]
        pruned.output_dim = pruned._compute_out_channels()
        pruned.out_channels = list(pruned.output_dim)
        pruned.embed_dim = list(pruned.output_dim)
        pruned.feature_fusion = nn.ModuleList([nn.Identity() for _ in pruned.output_dim])
        return pruned

    def encoders_info(self) -> list[dict]:
        """Describe specialists without interpreting or relabeling metadata."""
        return [dict(name=name, input_bands=ch, training_bands=list(bands), output_channels=list(out))
                for name, ch, bands, out in zip(self.encoder_names, self.encoders_input_bands,
                                              self.encoders_training_bands, self.encoders_out_channels)]

    def get_encoder_weights(self) -> torch.Tensor | None:
        if self.encoder_selector is not None:
            return self.encoder_selector.specialists_weights
        return None

    def get_fusion_l1_loss(self) -> torch.Tensor:
        if self.feature_fusion_mode != "conv1x1" or not self.regularize_fusion:
            return torch.tensor(0.0, device=self.device)
        l1 = torch.tensor(0.0, device=self.device)
        for layer in self.feature_fusion:
            if isinstance(layer, nn.Conv2d):
                l1 = l1 + torch.sum(torch.abs(layer.weight))
                if layer.bias is not None:
                    l1 = l1 + torch.sum(torch.abs(layer.bias))
        return l1

    # ---- Logging ---------------------------------------------------------------

    def _log_parameter_counts(self) -> None:
        def _count(m: nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        def _fmt(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.2f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)

        lines = ["\n" + "=" * 60, "EoS-FM parameter breakdown", "=" * 60]
        grand_total = grand_trainable = 0

        # Sub-encoders
        enc_total = enc_train = 0
        for name, enc in zip(self.encoder_names, self.encoders):
            t, tr = _count(enc)
            enc_total += t
            enc_train += tr
            lines.append(f"  Sub-encoder [{name:30s}] : {_fmt(t):>8s}  ({_fmt(tr)} trainable)")
        lines.append(f"  {'Sub-encoders TOTAL':40s} : {_fmt(enc_total):>8s}  ({_fmt(enc_train)} trainable)")
        grand_total += enc_total
        grand_trainable += enc_train

        for label, mod, cond in [
            ("Normalization layers", self.feature_normalizers, self.normalize_features),
            ("Projection layers", self.feature_projections, self.projection_layer),
        ]:
            if cond:
                t, tr = _count(mod)
                grand_total += t
                grand_trainable += tr
                lines.append(f"  {label:40s} : {_fmt(t):>8s}  ({_fmt(tr)} trainable)")

        if self.encoder_selector is not None:
            t, tr = _count(self.encoder_selector)
            grand_total += t
            grand_trainable += tr
            lines.append(f"  {'Encoder selector':40s} : {_fmt(t):>8s}  ({_fmt(tr)} trainable)")

        t, tr = _count(self.feature_fusion)
        grand_total += t
        grand_trainable += tr
        label = f"Feature fusion [{self.feature_fusion_mode or 'identity'}]"
        lines.append(f"  {label:40s} : {_fmt(t):>8s}  ({_fmt(tr)} trainable)")

        lines.append("-" * 60)
        lines.append(f"  {'GRAND TOTAL':40s} : {_fmt(grand_total):>8s}  ({_fmt(grand_trainable)} trainable)")
        lines.append("=" * 60)
        getLogger(__name__).debug("\n".join(lines))
