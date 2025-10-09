# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import PIL.Image
import time
import torch
import numpy as np
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output
import os

def print_banner(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../assets/owl_glove_small.jpg")
    parser.add_argument("--prompt", type=str, default="[an owl, a glove]")
    parser.add_argument("--threshold", type=str, default="0.1,0.1")
    parser.add_argument("--output", type=str, default="../data/owl_predict_out.jpg")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="../data/owl_image_encoder_patch32.engine")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_profiling_runs", type=int, default=30)
    args = parser.parse_args()

    # Clean prompt/threshold parsing
    prompt = args.prompt.strip("][()")
    text = [t.strip() for t in prompt.split(',')]
    thresholds = args.threshold.strip("][()").split(',')
    thresholds = float(thresholds[0]) if len(thresholds) == 1 else [float(x) for x in thresholds]

    # Load model
    print_banner("Initializing Model")
    # print(f"🔹 Model: {args.model}")
    predictor = OwlPredictor(args.model, image_encoder_engine=args.image_encoder_engine)

    # Load image
    image_pil = PIL.Image.open(args.image)
    image_w, image_h = image_pil.size
    # print(f"\nInput Image: {os.path.basename(args.image)}  ({image_w}x{image_h})")

    # Encode text prompt
    text_encodings = predictor.encode_text(text)

    # Measure inference time
    print_banner("Running Inference")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    output = predictor.predict(
        image=image_pil,
        text=text,
        text_encodings=text_encodings,
        threshold=thresholds,
        pad_square=False
    )
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    inference_time = (t_end - t_start) * 1000  # ms
    fps = 1.0 / (t_end - t_start)

    # Optional profiling
    if args.profile:
        torch.cuda.current_stream().synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(args.num_profiling_runs):
            _ = predictor.predict(
                image=image_pil,
                text=text,
                text_encodings=text_encodings,
                threshold=thresholds,
                pad_square=False
            )
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter_ns()
        dt = (t1 - t0) / 1e9
        print(f"\nProfiling FPS (avg over {args.num_profiling_runs} runs): {args.num_profiling_runs / dt:.2f}")

    # Draw detections
    image_np = np.array(image_pil)
    if not image_np.flags.writeable:
        image_np = image_np.copy()
    image_np = draw_owl_output(image_np, output, text=text, draw_text=True)
    out_path = args.output
    PIL.Image.fromarray(image_np).save(out_path)

    # Print result summary
    print_banner("Inference Summary")
    print(f"Model:            {args.model}")
    print(f"Image Enc. Engine: {args.image_encoder_engine}")
    print(f"Prompt:           {text}")
    print(f"Threshold(s):     {thresholds}")
    print(f"Input Image:      {os.path.basename(args.image)}  ({image_w}x{image_h})")
    print(f"Inference Time:   {inference_time:.2f} ms")
    print(f"Throughput:       {fps:.2f} FPS")
    print(f"Saved Output:     {out_path}")
    print("=" * 70 + "\n")
