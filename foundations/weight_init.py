import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return weights.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return weights.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Set seed ONCE at the start so the RNG stream flows continuously
        
        # 1. Build weight matrices first
        weights = []
        current_in = input_dim
        
        for _ in range(num_layers):
            if init_type == "xavier":
                std = math.sqrt(2.0 / (current_in + hidden_dim))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / current_in)
            elif init_type == "random":
                std = 1.0
            else:
                std = 1.0
                
            w = torch.randn(hidden_dim, current_in) * std
            weights.append(w)
            current_in = hidden_dim
            
        # 2. Forward a random input through them
        x = torch.randn(1, input_dim)
        stds = []
        
        # 3. Linear + ReLU per layer
        for w in weights:
            x = torch.matmul(x, w.t())
            x = torch.relu(x)
            
            # Record standard deviation rounded to 2 decimals
            stds.append(x.std().item())
            
        return stds
