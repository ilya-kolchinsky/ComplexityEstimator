import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hf.complexity_estimator import PromptComplexityConfig, PromptComplexityModel


def remap_state_dict(old_sd: dict) -> dict:
    new_sd = {}
    for k, v in old_sd.items():
        if k.startswith("enc.backbone."):
            new_k = "encoder." + k[len("enc.backbone."):]
        elif k.startswith("enc.post_ln."):
            new_k = "post_ln." + k[len("enc.post_ln."):]
        else:
            # head.*, proj.* should match; enc.* should have been handled above
            new_k = k
        new_sd[new_k] = v
    return new_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to the .pt file to convert")
    ap.add_argument("--out", required=True, help="Output folder for HF repo files")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=0, help="0 means auto(default)")
    ap.add_argument("--use_projection", action="store_true")
    ap.add_argument("--proj_hidden_ratio", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PromptComplexityConfig(
        base_model_name="microsoft/deberta-v3-base",
        max_length=args.max_length,
        dropout=args.dropout,
        hidden=None if args.hidden == 0 else args.hidden,
        layernorm_after_pool=True,
        use_projection=args.use_projection,
        proj_hidden_ratio=args.proj_hidden_ratio,
        output_sigmoid=True,
    )
    config.auto_map = {
        "AutoConfig": "complexity_estimator.configuration_prompt_complexity.PromptComplexityConfig",
        "AutoModel": "complexity_estimator.modeling_prompt_complexity.PromptComplexityModel",
    }

    model = PromptComplexityModel(config)

    old_sd = torch.load(args.pt, map_location="cpu")
    new_sd = remap_state_dict(old_sd)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Save weights/config in HF format
    model.save_pretrained(out_dir, safe_serialization=True)
    config.save_pretrained(out_dir)

    # Save tokenizer alongside for best UX
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    tok.save_pretrained(out_dir)

    print(f"Saved HF-ready model to: {out_dir}")


if __name__ == "__main__":
    main()
