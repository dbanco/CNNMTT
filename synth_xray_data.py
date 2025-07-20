# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:00:57 2025
@author: dpqb1
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def gaussian_basis_1d(N, mu, sigma, scaling='2-norm'):
    """
    Generate a wrapped 1D Gaussian basis vector of length N centered at mu.

    Args:
        N (int): Vector length
        mu (float): Mean (can be fractional, 0-based index)
        sigma (float): Standard deviation
        scaling (str): Scaling mode - '2-norm', '1-norm', 'max', 'rms', or 'pdf'

    Returns:
        numpy.ndarray: Vector of shape (N,)
    """
    idx = np.arange(N)

    # Circular distance
    dist = np.minimum(np.abs(idx - mu), N - np.abs(idx - mu))
    dist_sq = dist**2

    b = np.exp(-dist_sq / (2 * sigma**2))

    # Apply scaling
    if scaling == '2-norm':
        b /= np.linalg.norm(b)
    elif scaling == '1-norm':
        b /= np.sum(np.abs(b))
    elif scaling == 'max':
        b /= np.max(b)
    elif scaling == 'rms':
        b /= np.sqrt(np.mean(b**2))
    elif scaling == 'pdf':  # consistent with normal pdf
        b /= (sigma * np.sqrt(2 * np.pi))
    else:
        raise ValueError(f"Unknown scaling mode: {scaling}")

    return b

def sample_peak_positions(H, W, num_spots, min_sep=15, p_close=0.1):
    coords = []

    while len(coords) < num_spots:
        # Sample a new candidate
        y = np.random.randint(0, H)
        x = np.random.randint(0, W)
        new = np.array([y, x])

        # Measure distance to existing
        if coords:
            dists = [np.linalg.norm(new - np.array(c)) for c in coords]
            too_close = any(d < min_sep for d in dists)

            if too_close and np.random.rand() > p_close:
                continue  # Reject this one unless lucky
        coords.append((y, x))
    
    return coords


def generate_mtt_dataset_multichannel_truth(shape=(32, 96, 30), num_spots=np.random.randint(2,6), num_outputs=4, noise=True, seed=None):
    """
    Generate synthetic MTT dataset:
    - V: blurred Gaussian spots + noise (observed)
    - U: 3-channel truth tensor per frame with intensity, width_x, width_y at spot locations
    """
    if seed is not None:
        np.random.seed(seed)

    H, W, T = shape
    V = np.zeros(shape)
    U = np.zeros((num_outputs, H, W, T))  # 4 channels

    t_vals = np.linspace(0, 1, T)
    t_quad = t_vals ** 2

    amps = 20 + 130* np.random.rand(num_spots)

    # base_pos_y = H * (0.3 + 0.4 * np.random.rand(num_spots))
    # base_pos_x = W * (0.1 + 0.8 * np.random.rand(num_spots))
    
    coords = sample_peak_positions(0.5*H, 0.5*W, num_spots)
    base_pos_y = 0.25*H + np.array([y for y, x in coords])
    base_pos_x = 0.15*W + np.array([x for y, x in coords])

    shift_amp_y = H * 0.005 * np.random.rand(num_spots)          # tiny vertical shift
    shift_amp_x = W * (0.02 + 0.06 * np.random.rand(num_spots))  # larger horizontal shift

    base_width_y = 1.5 + 0.200 * np.random.rand(num_spots)
    base_width_x = 0.5 + 0.125 * np.random.rand(num_spots)

    grow_width_y = 0.1 + 0.3 * np.random.rand(num_spots)
    grow_width_x = 5   + 2   * np.random.randn(num_spots)

    pos_y_t = np.zeros((num_spots, T))
    pos_x_t = np.zeros((num_spots, T))
    width_y_t = np.zeros((num_spots, T))
    width_x_t = np.zeros((num_spots, T))

    psf_sigma = 1
    g_y = gaussian_basis_1d(H, H/2, psf_sigma, scaling='max')
    g_x = gaussian_basis_1d(W, W/2, psf_sigma, scaling='max')
    target_psf = np.outer(g_y, g_x)

    for i in range(num_spots):
        pos_y_t[i, :] = base_pos_y[i] + shift_amp_y[i] * t_quad
        pos_x_t[i, :] = base_pos_x[i] + shift_amp_x[i] * t_quad
        width_y_t[i, :] = base_width_y[i] + grow_width_y[i] * t_quad
        width_x_t[i, :] = base_width_x[i] + grow_width_x[i] * t_quad

    for t in range(T):
        frame_clean = np.zeros((H, W))
        frame_truth = np.zeros((4, H, W))

        for i in range(num_spots):
            # Build blurred Gaussian spot for observed data
            g_y = gaussian_basis_1d(H, pos_y_t[i, t], width_y_t[i, t])
            g_x = gaussian_basis_1d(W, pos_x_t[i, t], width_x_t[i, t])
            spot = amps[i] * np.outer(g_y, g_x)
            frame_clean += spot

            # Fill ground truth multi-channel tensor at nearest pixel
            py = int(np.clip(round(pos_y_t[i, t]), 0, H - 1))
            px = int(np.clip(round(pos_x_t[i, t]), 0, W - 1))

            frame_truth[0, py, px] += 100                # indicator
            frame_truth[1, py, px] += width_x_t[i, t]  # horizontal width
            frame_truth[2, py, px] += width_y_t[i, t]  # vertical width
            frame_truth[3, py, px] += amps[i]          # intensity
            
        if num_outputs == 1:
            frame_truth = convolve(frame_truth[0,:,:], target_psf, mode='constant')
            U[0, :, :, t] = frame_truth
        else: 
            # Normalize output channels
            frame_truth[0,:,:] = convolve(frame_truth[0,:,:], target_psf, mode='constant')
            frame_truth[1,:,:] = convolve(frame_truth[1,:,:]/20, target_psf, mode='constant') 
            frame_truth[2,:,:] = convolve(frame_truth[2,:,:]/1.4, target_psf, mode='constant') 
            frame_truth[3,:,:] = convolve(frame_truth[3,:,:]/150, target_psf, mode='constant') 
            U[:, :, :, t] = frame_truth
        
        V[:, :, t] = np.random.poisson(frame_clean) if noise else frame_clean

    spot_params = {
        'positions_y': pos_y_t,
        'positions_x': pos_x_t,
        'widths_y': width_y_t,
        'widths_x': width_x_t,
        'amplitudes': amps
    }

    return V, U, spot_params


class MTTSyntheticDataset(Dataset):
    def __init__(self, num_samples, sequence_length, input_shape, generate_fn, start_idx):
        """
        Args:
            num_samples (int): how many samples (time points) to generate
            input_shape (tuple): (C_in, H, W) shape of input observations
            generate_fn (function): user-defined function that returns
                (input_obs, truth_mask) tuple of numpy arrays
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.generate_fn = generate_fn
        self.start_idx = start_idx

    def __len__(self):
        return self.num_samples * self.sequence_length

    def __getitem__(self, idx):
        # Flattened index -> (sequence_index, frame_index)
        seq_idx = self.start_idx + (idx // self.sequence_length)
        frame_idx = idx % self.sequence_length

        # Generate or retrieve the (seq_idx)-th sequence, and pick frame_idx
        input_img, truth_mask = self.generate_fn(seq_idx, frame_idx)
        return input_img.float(), truth_mask.float()

# Global caches
V_global = {}
U_global = {}

# This function adapts the big generated volume to one sample per call for Dataset
def generate_fn(seq_idx, frame_idx):
    # Seed based on sequence so we can have diverse tracks per sequence
    np.random.seed(seq_idx)

    # Generate a sequence of simulated targets
    V_global[seq_idx], U_global[seq_idx], _ = generate_mtt_dataset_multichannel_truth(seed=seq_idx)

    input_img = V_global[seq_idx][:, :, frame_idx]  # shape (H, W)
    truth_mask = U_global[seq_idx][:, :, :, frame_idx]  # shape (4, H, W)

    input_img = torch.from_numpy(input_img).unsqueeze(0)  # (1, H, W)
    truth_mask = torch.from_numpy(truth_mask)             # (4, H, W)
    return input_img, truth_mask
    

if __name__ == "__main__": 
    
    # Generate the full dataset once for the sake of example
    V, U, params = generate_mtt_dataset_multichannel_truth(shape=(32, 96, 30), seed=np.random.randint(0,10000))

    fig, axs = plt.subplots(5, 4, figsize=(12, 8), constrained_layout=True)
    
    # Example: visualize intensity channel and widths at several frames
    for f,frame in enumerate([0, 5, 10, 20, 29]):  # example time frame to visualize

        titles = ['Truth Intensity (Channel 0)', 'Truth Width X (Channel 1)', 'Truth Width Y (Channel 2)', 'Observed Data V']
    
        # Plot truth channels
        for i in range(3):
            im = axs[f][i].imshow(U[i, :, :, frame], cmap='hot', aspect='auto')
            axs[f][i].set_title(titles[i])
            axs[f][i].axis('off')
            # fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    
        # Plot observed data
        im = axs[f][3].imshow(V[:, :, frame], cmap='hot', aspect='auto')
        axs[f][3].set_title(titles[3])
        axs[f][3].axis('off')
        fig.colorbar(im, ax=axs[f][3], fraction=0.046, pad=0.04)
    
        plt.show()

