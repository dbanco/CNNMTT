# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 23:04:47 2025

@author: dpqb1
"""

import torch
from mtt_cnn import MTTModelQAT  # or your model class
from torch.ao.quantization import prepare_qat, convert, get_default_qat_qconfig
from torch.quantization.qconfig import QConfig
from torch.ao.quantization.observer import default_observer, default_weight_observer
import time

def load_quantized_model(model_path):
    model = MTTModelQAT(input_channels=1, output_channels=1)  # match your architecture
    model.eval()
    
    # Now fuse modules and prepare QAT
    model.fuse_model()  # fuse conv+relu pairs
    # qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    
    # Use per-tensor for both activations and weights
    qat_qconfig = QConfig(
        activation=default_observer,
        weight=default_weight_observer
    )
    model.qconfig = qat_qconfig
    return model

def run_inference(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return output

if __name__ == "__main__":
    model_path = "qat_quantized_model.pth"
    model = load_quantized_model(model_path)
    
    # Dummy input example: adjust to your input shape
    dummy_input = torch.randn(2000, 1, 32, 96)
    
    start = time.time()
    output = run_inference(model, dummy_input)
    print(f"Elapsed time: {time.time() - start:.3f} seconds")
