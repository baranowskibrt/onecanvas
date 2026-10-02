"""Inspect the shared loader's derived fixed-to-free gate scale on CPU.

Run with the stage-1 checkpoint directory. Prints only the scale to stdout.
Training does not call this script; the maintained stage-2 loader owns and
records the actual conversion.
"""
import os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_adapters.geometry_embedding import GeometryEmbeddingMixin
from utils.embedding_io import (_calibrated_gate_scale, _source_config,
                                _source_fixed_ratio,
                                _validate_calibration_metadata)
ckpt = sys.argv[1]
cfg, cfg_path = _source_config(ckpt)
target = _source_fixed_ratio(cfg, cfg_path)
if target <= 0:
    raise SystemExit(f"Source is already free-running (fixed ratio {target})")
data = cfg["data"]
sd = torch.load(os.path.join(ckpt, "depth_embedding.pt"),
                map_location="cpu", weights_only=True)
mlp_state = sd["cartesian_fourier_mlp"]
hidden_size = int(mlp_state["2.weight"].shape[0])
class CalibrationEmbedding(GeometryEmbeddingMixin, torch.nn.Module):
    pass
m = CalibrationEmbedding()
m.init_depth_embedding(hidden_size=hidden_size,
                       num_freqs=int(data["depth_embed_num_freqs"]),
                       mlp_hidden=int(data["depth_embed_mlp_hidden"]),
                       cartesian_fourier_use_rmsnorm=bool(data[
                           "depth_embed_cartesian_fourier_use_rmsnorm"]),
                       cartesian_fourier_per_channel_gate=bool(data[
                           "depth_embed_cartesian_fourier_per_channel_gate"]),
                       cartesian_fourier_gate_init=1.0,
                       cartesian_fourier_mlp_init_std=0.06)
m.depth_cartesian_fourier_mlp.load_state_dict(sd["cartesian_fourier_mlp"])
with torch.no_grad():
    m.depth_cartesian_fourier_gate.copy_(sd["cartesian_fourier_gate"])
    m._depth_fourier_freqs.copy_(sd["cartesian_fourier_freqs"])
if hasattr(m, "depth_cartesian_fourier_norm"):
    m.depth_cartesian_fourier_norm.load_state_dict(sd["cartesian_fourier_norm"])
_validate_calibration_metadata(m, cfg, cfg_path)
fold, raw = _calibrated_gate_scale(m, target)
gate = float(m.depth_cartesian_fourier_gate.float().mean())
print(f"[gate-fold] stage-1 pin {target}  saved gate {gate:.4f}  raw ratio {raw:.3f}  "
      f"gate x {fold:.5f}  -> gate {gate * fold:.4f}", file=sys.stderr)
print(f"{fold:.6f}")
