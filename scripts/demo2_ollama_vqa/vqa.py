#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Consistent, bannered printing + Ollama token/time stats

import argparse
import time
import os
import tempfile
from pathlib import Path

from PIL import Image  # pip install pillow
import ollama


# --------------------------- Utils ---------------------------

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def round_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return max(1, x)
    return max(m, int(round(x / m) * m))

def ns_to_ms(x: int | None) -> float | None:
    if x is None:
        return None
    return float(x) / 1e6

def pretty_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def resize_for_vlm(
    in_path: str,
    long_side: int = 1024,
    stride: int = 28,
    max_long_side: int | None = None,
    out_suffix: str = "_resized",
) -> tuple[str, tuple[int, int], float]:
    """
    Resize image to have a fixed long side, keep aspect ratio,
    then round width & height to a given stride (e.g., 28).
    Returns (out_path, (w,h), scale) where scale is new_long_side/original_long_side.
    """
    img = Image.open(in_path).convert("RGB")
    w, h = img.size

    if max_long_side is not None:
        long_side = min(long_side, max_long_side)

    if max(w, h) == 0:
        raise ValueError("Invalid image with zero dimension.")

    scale = float(long_side) / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if stride and stride > 1:
        new_w = round_to_multiple(new_w, stride)
        new_h = round_to_multiple(new_h, stride)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    resized = img.resize((new_w, new_h), resample=Image.LANCZOS)

    base = Path(in_path).stem
    tmpdir = tempfile.mkdtemp(prefix="vlm_preproc_")
    out_path = os.path.join(tmpdir, f"{base}{out_suffix}_{new_w}x{new_h}.png")
    resized.save(out_path, format="PNG", optimize=True)
    return out_path, (new_w, new_h), scale


# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run an Ollama VLM (e.g., qwen2.5vl:3b) with image + query, with pre-resize and stats."
    )
    parser.add_argument("--model", type=str, default="qwen2.5vl:3b",
                        help="Model name (default: qwen2.5vl:3b)")
    parser.add_argument("--image", type=str, default="./golf.png",
                        help="Path to input image")
    parser.add_argument("--query", type=str,
                        default="What sport is this?",
                        help="Text prompt / question")
    parser.add_argument("--long-side", type=int, default=512,
                        help="Resize so the long side equals this value (default: 512)")
    parser.add_argument("--stride", type=int, default=28,
                        help="Round width/height to a multiple of this value (default: 28). Use 1 to disable.")
    parser.add_argument("--no-round", action="store_true",
                        help="Disable rounding to stride (overrides --stride).")
    parser.add_argument("--max-long-side", type=int, default=None,
                        help="Optional upper cap for long side (e.g., 1280).")
    parser.add_argument("--show-raw", action="store_true",
                        help="Print the full raw response dict from Ollama.")
    args = parser.parse_args()

    # -------------------- Initializing Model (metadata) --------------------
    print_banner("Initializing Model")
    print(f"Model:             {args.model}")
    in_name = os.path.basename(args.image)
    try:
        _w, _h = Image.open(args.image).size
        print(f"Input Image:       {in_name}  ({_w}x{_h})")
    except Exception:
        print(f"Input Image:       {in_name}  (size: unknown)")
    print(f"Query:             {args.query}")

    # -------------------- Preprocess --------------------

    stride = 1 if args.no_round else max(1, args.stride)
    t_pp0 = time.perf_counter()
    pre_img_path, (new_w, new_h), scale = resize_for_vlm(
        args.image,
        long_side=args.long_side,
        stride=stride,
        max_long_side=args.max_long_side,
    )
    t_pp1 = time.perf_counter()
    mpix = (new_w * new_h) / 1e6
    fsize = os.path.getsize(pre_img_path)

    # -------------------- Running Inference --------------------
    print_banner("Running Inference")
    t0 = time.perf_counter()
    response = ollama.chat(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": args.query,
                "images": [pre_img_path],
            }
        ],
    )
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0

    # Parse response
    msg = response.get("message", {})
    content = msg.get("content", "")

    prompt_tok = response.get("prompt_eval_count")          # input tokens
    out_tok = response.get("eval_count")                    # output tokens
    total_ns = response.get("total_duration")
    prompt_ns = response.get("prompt_eval_duration")
    gen_ns = response.get("eval_duration")
    done_reason = response.get("done_reason")

    print(f"Model Output:\n{content}")

    # -------------------- Inference Summary --------------------
    print_banner("Inference Summary")
    print(f"Model:             {args.model}")
    print(f"Query:             {args.query}")
    print(f"Input Image:       {os.path.basename(args.image)}")
    print(f"Resized Dims:      {new_w} x {new_h}")
    # print(f"Wall Time:         {wall_ms:.2f} ms")

    if total_ns is not None:
        print(f"total_duration:    {ns_to_ms(total_ns):.1f} ms")
    if prompt_ns is not None:
        print(f"prompt_eval_dur:   {ns_to_ms(prompt_ns):.1f} ms")
    if gen_ns is not None:
        print(f"eval_duration:     {ns_to_ms(gen_ns):.1f} ms")
    if prompt_tok is not None:
        print(f"Input Tokens:      {prompt_tok}")
    if out_tok is not None:
        print(f"Output Tokens:     {out_tok}")
    print("=" * 70 + "\n")

    if args.show_raw:
        import json
        print_banner("Raw Response (Ollama)")
        print(json.dumps(response, indent=2))
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
