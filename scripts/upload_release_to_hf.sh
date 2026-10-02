#!/bin/bash
# Upload the merged release checkpoint (and the stage-1 adapter) to Hugging Face.
#
# One-time setup:
#   pip install -U huggingface_hub
#   hf auth login              # paste a WRITE-scoped token from hf.co/settings/tokens
#
# Usage:
#   bash scripts/upload_release_to_hf.sh <username-or-org>/OneCanvas-Qwen3-VL-8B <merged-dir> <stage1-dir>
#
# The repo is created PRIVATE. Flip it public on hf.co/<repo>/settings after
# download-testing it.
set -euo pipefail

REPO_ID=${1:?Set the Hugging Face repository id}
MERGED=${2:?Set the verified merged checkpoint directory}
STAGE1=${3:?Set the matching stage-1 adapter directory}

# PREFLIGHT. The 2026-07-09 upload shipped a config.json written by transformers
# 5.2.0, which spells the rope block `rope_parameters`. On the 4.57.x floor
# pyproject declares, the model reads `rope_scaling`, gets None, and dies before
# loading a single sample. It was published broken and stayed that way until
# 2026-08-21, so the check runs here, at the publish gate.
python - "$MERGED" <<'EOF'
import json, sys
from pathlib import Path

merged = Path(sys.argv[1])
fail = []

if "qwen3-vl" not in merged.name.lower() and "qwen3_vl" not in merged.name.lower():
    # _is_qwen3_vl() routes the model class by substring-matching the path
    # name; a miss silently falls through to Qwen3.5.
    fail.append(f"dir name '{merged.name}' must contain 'Qwen3-VL'")

cfg = json.load(open(merged / "config.json"))
node = cfg.get("text_config", cfg)
scaling, params = node.get("rope_scaling"), node.get("rope_parameters")
if not scaling:
    fail.append("text_config.rope_scaling missing, transformers 4.57.x cannot load this")
if not params:
    fail.append("text_config.rope_parameters missing, transformers 5.x cannot load this")
if scaling and not scaling.get("mrope_section"):
    fail.append("rope_scaling.mrope_section missing")
if node.get("rope_theta") is None:
    fail.append("text_config.rope_theta missing, 4.57.x would use the class default")

tok = json.load(open(merged / "tokenizer_config.json"))
if isinstance(tok.get("extra_special_tokens"), list):
    # 5.x spelling; 4.57.x feeds it to _set_model_specific_special_tokens and
    # dies with "'list' object has no attribute 'keys'".
    fail.append("tokenizer_config.extra_special_tokens is a list, "
                "transformers 4.57.x cannot load this")
if not tok.get("additional_special_tokens"):
    fail.append("tokenizer_config.additional_special_tokens missing, "
                "the vision/control tokens would not register as special")
for key in ("backend", "is_local"):
    if key in tok:
        fail.append(f"tokenizer_config.{key} is a 5.x-only field, "
                    f"4.57.x forwards it into the tokenizer constructor")

# The processor must actually build on the declared floor, not merely parse.
try:
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(str(merged))
except Exception as e:
    fail.append(f"AutoProcessor.from_pretrained failed here: {type(e).__name__}: {e}")

stray = sorted(p.name for p in merged.iterdir()
               if ".bak" in p.name or p.suffix == ".tmp" or p.name.endswith("~"))
if stray:
    fail.append(f"stray files would be published: {', '.join(stray)}")

for name in ("resolved_config.json", "depth_embedding.pt"):
    if not (merged / name).exists():
        fail.append(f"{name} missing, the public code needs it")

if fail:
    print("PREFLIGHT FAILED, nothing uploaded:")
    for f in fail:
        print(f"  - {f}")
    print("\nFor the rope and tokenizer keys, re-run the exporter or apply "
          "ensure_cross_version_rope_config() / ensure_cross_version_tokenizer_config() "
          "from scripts/export_merged_checkpoint.py")
    sys.exit(1)
print(f"[preflight] {merged.name} OK")
EOF

# Create the repo (no-op if it exists) and a stage1 branch.
python - "$REPO_ID" <<'EOF'
import sys
from huggingface_hub import HfApi
api = HfApi()
repo_id = sys.argv[1]
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
try:
    api.create_branch(repo_id, branch="stage1", exist_ok=True)
except Exception as e:
    print(f"branch: {e}")
print(f"repo ready: https://huggingface.co/{repo_id}")
EOF

# Main branch: the merged model. upload-large-folder is resumable, so a
# dropped connection continues instead of restarting 17 GB.
hf upload-large-folder "$REPO_ID" "$MERGED" --repo-type model

# stage1 branch: adapter + 3D position embedding only (small). Only the
# files stage-2 retraining needs.
hf upload "$REPO_ID" "$STAGE1/adapter_config.json" adapter_config.json --revision stage1
hf upload "$REPO_ID" "$STAGE1/adapter_model.safetensors" adapter_model.safetensors --revision stage1
hf upload "$REPO_ID" "$STAGE1/depth_embedding.pt" depth_embedding.pt --revision stage1

echo "Done. Verify with a fresh download before flipping public:"
echo "  hf download $REPO_ID --local-dir /tmp/hf_check"
