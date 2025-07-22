# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 21:29:43 2025

@author: dpqb1
"""
# fast_generate_mtt_dataset.py

import torch
import os
import time
from synth_xray_data import MTTSyntheticDataset

def save_preloaded_dataset(
    output_path="dataset.pt",
    num_samples=30, 
    sequence_length=30,
    input_shape=(1, 32, 96),
    seed=42,
    num_spots=3,
    noise=True
):
    start = time.time()
    print(f"[INFO] Generating {num_samples} sequences of {sequence_length} frames...")

    dataset = MTTSyntheticDataset(
        num_spots=num_spots,
        num_samples=num_samples,
        sequence_length=sequence_length,
        noise=noise,
        input_shape=input_shape,
        seed=seed,
        preload=True   # preload all at once
    )

    # Convert to contiguous tensors if needed
    inputs = dataset.inputs
    labels = dataset.labels

    data = {
        "inputs": inputs,      # shape: (N, T, C, H, W)
        "labels": labels       # shape: (N, T, C, H, W) or whatever label format
    }

    print(f"[INFO] Saving to {output_path}...")
    torch.save(data, output_path)
    print(f"[DONE] Saved {output_path} in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="mtt_dataset.pt")
    parser.add_argument("--N", type=int, default=30000)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--W", type=int, default=96)
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spots", type=int, default=3)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    save_preloaded_dataset(
        output_path=args.out,
        num_samples=args.N,
        sequence_length=args.T,
        input_shape=(args.C, args.H, args.W),
        seed=args.seed,
        num_spots=args.spots,
        noise=args.noise
    )
