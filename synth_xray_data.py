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
import torch.distributed as dist
import time
import concurrent.futures
from tqdm import tqdm


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
        norm = np.linalg.norm(b)
        if norm > 0:
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

def generate_mtt_dataset_multichannel_truth(shape=(32, 96, 30), num_spots=np.random.randint(2,6), num_outputs=1, noise=True, seed=None):
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
        
    V = np.expand_dims(V, axis=0)

    spot_params = {
        'positions_y': pos_y_t,
        'positions_x': pos_x_t,
        'widths_y': width_y_t,
        'widths_x': width_x_t,
        'amplitudes': amps
    }

    return V, U, spot_params

class MTTSyntheticDataset(Dataset):
    def __init__(self, num_spots, num_samples, sequence_length, noise, input_shape, seed=None, preload=False):
        """
        num_samples: Number of synthetic sequences to generate
        sequence_length: Frames per sequence
        input_shape: (C, H, W)
        generate_fn: Function that generates (input_seq, truth_seq) given seed/index
        start_idx: Offset for seeding sequence generation
        cache: Whether to precompute and store full sequences
        """
        self.num_spots = num_spots
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.noise = noise
        self.preload = preload
        
        if seed is not None:
            np.random.seed(seed)

        self.total_frames = num_samples * sequence_length
        self.precompute_params()
        
        if self.preload:
            self.preload_all()

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if self.preload:
            return self.inputs[idx], self.labels[idx]
        else:
            seq_idx = idx // self.sequence_length
            frame_idx = idx % self.sequence_length
            try:
                input_img, truth_mask = self.generate_frame(seq_idx, frame_idx)
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                return None
            return torch.tensor(input_img, dtype=torch.float32), torch.tensor(truth_mask, dtype=torch.float32)



    def precompute_params(self):
        T = self.sequence_length
        C, H, W = self.input_shape
        t_vals = np.linspace(0, 1, T)
        self.t_quad = t_vals ** 2
        
        self.amps = 20 + 130* np.random.rand(self.num_samples,self.num_spots)
        self.base_pos_y = H*(0.25 + 0.5*np.random.rand(self.num_samples,self.num_spots))
        self.base_pos_x = W*(0.15 + 0.5*np.random.rand(self.num_samples,self.num_spots))

        self.shift_amp_y = H * 0.005 * np.random.rand(self.num_samples,self.num_spots)      # tiny vertical shift
        self.shift_amp_x = W * (0.02 + 0.06 * np.random.rand(self.num_samples,self.num_spots))  # larger horizontal shift

        self.base_width_y = 1.5 + 0.200 * np.random.rand(self.num_samples,self.num_spots)
        self.base_width_x = 0.5 + 0.125 * np.random.rand(self.num_samples,self.num_spots)

        self.grow_width_y = 0.1 + 0.3 * np.random.rand(self.num_samples,self.num_spots)
        self.grow_width_x = 5   + 2   * np.random.randn(self.num_samples,self.num_spots)**2

        psf_sigma = 1
        g_y = gaussian_basis_1d(H, H/2, psf_sigma, scaling='max')
        g_x = gaussian_basis_1d(W, W/2, psf_sigma, scaling='max')
        self.target_psf = np.outer(g_y, g_x)
        
    def generate_frame(self, seq_idx, t):
        i = seq_idx
        C, H, W = self.input_shape
        frame_clean = np.zeros((H, W))
        frame_truth = np.zeros((H, W))

        V = np.zeros((1, H, W))  
        U = np.zeros((1, H, W))  
        
        for j in range(self.num_spots):
            pos_y_t = self.base_pos_y[i,j] + self.shift_amp_y[i,j] * self.t_quad[t]
            pos_x_t = self.base_pos_x[i,j] + self.shift_amp_x[i,j] * self.t_quad[t]
            width_y_t = self.base_width_y[i,j] + self.grow_width_y[i,j] * self.t_quad[t]
            width_x_t = self.base_width_x[i,j] + self.grow_width_x[i,j] * self.t_quad[t]
             
            if width_y_t <= 0 or width_x_t <= 0:
                raise ValueError(f"Invalid width: width_y={width_y_t}, width_x={width_x_t}")
           
            g_y = gaussian_basis_1d(H, pos_y_t, width_y_t)
            g_x = gaussian_basis_1d(W, pos_x_t, width_x_t)
            spot = self.amps[i,j] * np.outer(g_y, g_x)
            if np.any(np.isnan(spot)) or np.any(np.isinf(spot)):
                raise ValueError(f"NaN/Inf in spot for i={i}, j={j}")
            
            frame_clean += spot
            
            py = int(np.clip(round(pos_y_t), 0, H - 1))
            px = int(np.clip(round(pos_x_t), 0, W - 1))

            frame_truth[py, px] += 100      # indicator

        frame_truth = convolve(frame_truth, self.target_psf, mode='constant')
        
        U[0, :, :] = frame_truth
        V[0, :, :] = np.random.poisson(frame_clean) if self.noise else frame_clean

        return V, U
    
    def preload_all(self):
        print(f"[INFO] Preloading {self.total_frames} frames in parallel...")
        indices = [(seq_idx, frame_idx)
                   for seq_idx in range(self.num_samples)
                   for frame_idx in range(self.sequence_length)]
    
        def gen_frame(idx_pair):
            seq_idx, frame_idx = idx_pair
            V, U = self.generate_frame(seq_idx, frame_idx)
            return torch.tensor(V, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            results = list(executor.map(gen_frame, indices))
    
        self.inputs, self.labels = zip(*results)
        # print(f"[INFO] Preloading complete.")
       

if __name__ == "__main__": 
    from torch.utils.data import DataLoader
    train_dataset = MTTSyntheticDataset(num_spots=3,
                                        num_samples=100,
                                        sequence_length=30,
                                        noise=True,
                                        input_shape=(1, 32, 96),
                                        seed=1,
                                        preload=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    start = time.time()   
    aa = 0
    for batch_idx, (inputs, truths) in enumerate(train_loader):
        aa+=1
    print(f"Avg sample generation time: {(time.time() - start) / 100:.4f} seconds")
    
    
    # plt.figure()
    # plt.imshow(inputs[0,0,:,:])

    # plt.figure()
    # plt.imshow(truths[0,0,:,:])
    
    
    '''
    # Generate the full dataset once for the sake of example
    V, U, params = generate_mtt_dataset_multichannel_truth(shape=(32, 96, 30), seed=np.random.randint(0,10000))

    fig, axs = plt.subplots(5, 4, figsize=(12, 8), constrained_layout=True)
    
    # Example: visualize intensity channel and widths at several frames
    for f,frame in enumerate([0, 5, 10, 20, 29]):  # example time frame to visualize

        titles = ['Truth Intensity (Channel 0)', 'Truth Width X (Channel 1)', 'Truth Width Y (Channel 2)', 'Observed Data V']
    
        # Plot truth channels
        for i in range(1):
            im = axs[f][i].imshow(U[i, :, :, frame], cmap='hot', aspect='auto')
            axs[f][i].set_title(titles[i])
            axs[f][i].axis('off')
            # fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    
        # Plot observed data
        im = axs[f][3].imshow(V[0, :, :, frame], cmap='hot', aspect='auto')
        axs[f][3].set_title(titles[3])
        axs[f][3].axis('off')
        fig.colorbar(im, ax=axs[f][3], fraction=0.046, pad=0.04)
    
        plt.show()
    '''
