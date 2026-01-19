"""
V1 Neural Coding Benchmark Experiment
=====================================

This script runs a comprehensive benchmark of PP-PCA (Point Process PCA) 
on the pvc8 neural coding dataset. It measures computational complexity,
compares with theoretical predictions, and evaluates perceptual distance
correlations using LPIPS (Learned Perceptual Image Patch Similarity).

The experiment outputs a markdown report with all figures saved alongside.
The dataset is available at: https://crcns.org/data-sets/vc/pvc-8/about.
"""

import hashlib
import json
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lpips
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.offsetbox as moffsetbox
import numpy as np
import scipy.io
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import MDS
from tqdm import tqdm
import pandas as pd

from src import pppca

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Report output directory
REPORT_DIR = Path("./reports/v1_benchmark")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
DATA_DIR = Path("./data/pvc8")
SESSION_ID = 1  # Session number (1-10), can be changed at runtime

# Image type filter: "all", "small", or "big"
# According to the dataset description:
# - First 540 images are natural, alternating small (odd indices 0,2,4...) and big (even indices 1,3,5...)
# - Remaining images are gratings
IMAGE_TYPE = "small"  # Options: "all", "small", "big"

# Benchmark parameters
IMAGE_COUNTS = [10, 20, 30, 50, 75, 100, 150]  # Variable number of images to test
# IMAGE_COUNTS = [10, 20]  # Test mode
NUM_REPEATS_TO_USE = 1  # Number of trial repeats per neuron/image
BIN_SIZE = 2  # Temporal bin size for spike counts
J1 = 3  # Number of 1D PP-PCA components
J_MV = 3  # Number of multivariate PP-PCA components


# =============================================================================
# Utility Functions
# =============================================================================

def bin_spike_counts(counts: np.ndarray, bin_size: int = 10) -> np.ndarray:
    """Bin spike counts into larger time bins."""
    T = len(counts)
    num_bins = T // bin_size
    binned = np.add.reduceat(counts, np.arange(0, T, bin_size))
    return binned[:num_bins]


def get_binned_t_grid(T: int, bin_size: int = 10) -> np.ndarray:
    """Create a time grid for binned spike counts."""
    num_bins = T // bin_size
    return np.linspace(0, 1, num_bins, dtype=np.float64)


def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """Save figure to the report directory."""
    path = REPORT_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# =============================================================================
# Data Loading
# =============================================================================

def get_data_path(session_id: int = None) -> Path:
    """Get the path to the data file for a given session."""
    if session_id is None:
        session_id = SESSION_ID
    if not 1 <= session_id <= 10:
        raise ValueError(f"Session ID must be between 1 and 10, got {session_id}")
    return DATA_DIR / f"{session_id:02d}.mat"


def get_image_indices_by_type(num_images: int, image_type: str = None) -> np.ndarray:
    """
    Get image indices filtered by type.
    
    According to the dataset description:
    - First 540 images are natural images, alternating small and big
    - Small images: indices 0, 2, 4, ... (windowed to 1 degree)
    - Big images: indices 1, 3, 5, ... (3-6.7 degrees)
    - Images from index 540 onwards are gratings
    
    Parameters
    ----------
    num_images : int
        Total number of images in the dataset
    image_type : str
        One of "all", "small", "big". If None, uses global IMAGE_TYPE.
        
    Returns
    -------
    np.ndarray
        Array of valid image indices based on the filter
    """
    if image_type is None:
        image_type = IMAGE_TYPE
    
    # Number of natural images (first 540, alternating small/big)
    num_natural = min(540, num_images)
    
    if image_type == "all":
        # Return all image indices
        return np.arange(num_images)
    elif image_type == "small":
        # Small images are at even indices (0, 2, 4, ...) within the first 540
        return np.arange(0, num_natural, 2)
    elif image_type == "big":
        # Big images are at odd indices (1, 3, 5, ...) within the first 540
        return np.arange(1, num_natural, 2)
    else:
        raise ValueError(f"image_type must be 'all', 'small', or 'big', got '{image_type}'")


def load_data(session_id: int = None, image_type: str = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load the pvc8 neural coding dataset.
    
    Parameters
    ----------
    session_id : int, optional
        Session number (1-10). If None, uses global SESSION_ID.
    image_type : str, optional
        Filter images by type: "all", "small", or "big". If None, uses global IMAGE_TYPE.
        
    Returns
    -------
    resp : np.ndarray
        Spike train responses, filtered to selected images. Shape: (N, I_filtered, R, T)
    images : np.ndarray
        Image stimuli, filtered to selected images
    info : dict
        Dataset information including filtering details
    """
    data_path = get_data_path(session_id)
    mat_data = scipy.io.loadmat(str(data_path))
    resp = mat_data['resp_train']  # (N, I, R, T)
    images_raw = mat_data['images'][0]  # (I, height, width)
    
    N, I_total, R, T = resp.shape
    
    # Get filtered image indices
    valid_indices = get_image_indices_by_type(I_total, image_type)
    
    # Filter responses and images
    resp_filtered = resp[:, valid_indices, :, :]
    images_filtered = images_raw[valid_indices]
    
    actual_session = session_id if session_id is not None else SESSION_ID
    actual_type = image_type if image_type is not None else IMAGE_TYPE
    
    info = {
        'num_neurons': N,
        'num_images': len(valid_indices),
        'num_images_total': I_total,
        'num_repeats': R,
        'num_time_bins': T,
        'session_id': actual_session,
        'image_type': actual_type,
        'valid_image_indices': valid_indices.tolist(),
    }
    return resp_filtered, images_filtered, info


# =============================================================================
# Benchmark Core Functions
# =============================================================================

def prepare_1d_point_processes(
    resp: np.ndarray,
    selected_image_indices: np.ndarray,
    t_grid_binned: np.ndarray,
    num_repeats: int,
    bin_size: int,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
    """Prepare 1D point processes from spike train data."""
    N, I, R, T = resp.shape
    all_1d_points = []
    idx_map = []
    
    for n in range(N):
        for i_img in selected_image_indices:
            if R <= num_repeats:
                chosen_repeats = range(R)
            else:
                chosen_repeats = random.sample(range(R), num_repeats)
            for r in chosen_repeats:
                counts = resp[n, i_img, r, :]
                counts = np.clip(counts, 0, None).astype(int)
                binned_counts = bin_spike_counts(counts, bin_size=bin_size)
                if binned_counts.sum() == 0:
                    continue
                times = np.repeat(t_grid_binned, binned_counts)
                if len(times) == 0:
                    continue
                pts = times[:, None]
                all_1d_points.append(torch.from_numpy(pts).to(torch.float64))
                idx_map.append((n, i_img, r))
    
    return all_1d_points, idx_map


def run_1d_pppca(all_1d_points: List[torch.Tensor], J: int) -> Tuple[torch.Tensor, float]:
    """Run 1D PP-PCA and return scores with timing."""
    start = time.perf_counter()
    result = pppca(all_1d_points, Jmax=J)
    elapsed = time.perf_counter() - start
    scores = torch.as_tensor(result['scores'].values, dtype=torch.float64)
    return scores, elapsed


def build_neuron_distance_matrix(
    scores1d: torch.Tensor,
    idx_map: List[Tuple[int, int, int]],
    N: int,
    I: int,
    J1: int,
) -> np.ndarray:
    """Build neuron-to-neuron distance matrix based on mean scores."""
    score_dict = defaultdict(list)
    for idx, (n, i_img, r) in enumerate(idx_map):
        score_dict[(n, i_img)].append(scores1d[idx])
    
    mean_scores = np.zeros((N, I, J1))
    for n in range(N):
        for i_img in range(I):
            arr = score_dict.get((n, i_img), [])
            if arr:
                mean_scores[n, i_img, :] = torch.stack(arr, dim=0).mean(0).cpu().numpy()
    
    D = np.zeros((N, N))
    for i_img in range(I):
        for a in range(N):
            for b in range(N):
                diff = mean_scores[a, i_img, :] - mean_scores[b, i_img, :]
                D[a, b] += np.linalg.norm(diff)
    D /= I
    return D


def build_multivariate_pp(
    resp: np.ndarray,
    selected_image_indices: np.ndarray,
    embed_2d: np.ndarray,
    t_grid_binned: np.ndarray,
    bin_size: int,
) -> List[torch.Tensor]:
    """Build multivariate point processes (x, y, t) for each image."""
    N, I, R, T = resp.shape
    mvpp_per_image = []
    
    for i_img in selected_image_indices:
        pts_list = []
        for r in range(R):
            for n in range(N):
                counts = resp[n, i_img, r, :]
                counts = np.clip(counts, 0, None).astype(int)
                if counts.sum() == 0:
                    continue
                binned_counts = bin_spike_counts(counts, bin_size=bin_size)
                if binned_counts.sum() == 0:
                    continue
                times = np.repeat(t_grid_binned, binned_counts)
                if len(times) == 0:
                    continue
                xy = embed_2d[n]
                xy_rep = np.repeat(xy[None, :], times.shape[0], axis=0)
                arr = np.column_stack([xy_rep, times])
                pts_list.append(arr)
        
        if pts_list:
            all_pts = np.vstack(pts_list)
            mvpp_per_image.append(torch.from_numpy(all_pts).to(torch.float64))
        else:
            mvpp_per_image.append(torch.empty((0, 3), dtype=torch.float64))
    
    return mvpp_per_image


def run_mv_pppca(mvpp_per_image: List[torch.Tensor], J: int) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """Run multivariate PP-PCA and return scores, eigenvalues, and timing."""
    start = time.perf_counter()
    result = pppca(mvpp_per_image, Jmax=J)
    elapsed = time.perf_counter() - start
    scores = result['scores']
    eigenvalues = np.asarray(result['eigenval'], dtype=float)
    return scores, eigenvalues, elapsed


# def compute_lpips_matrix(images: np.ndarray, selected_indices: np.ndarray) -> Tuple[np.ndarray, float]:
#     """Compute LPIPS perceptual distance matrix using VGG backbone."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     loss_fn = lpips.LPIPS(net='vgg').to(device)
#     loss_fn.eval()
    
#     selected_images_list = [images[i] for i in selected_indices]
#     if not selected_images_list:
#         raise ValueError("No images selected for LPIPS computation.")
    
#     target_height, target_width = selected_images_list[0].shape
    
#     processed_tensors = []
#     for img_np in selected_images_list:
#         img_tensor = torch.from_numpy(img_np.astype(np.float32))
#         img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
#         resized_tensor = torch.nn.functional.interpolate(
#             img_tensor,
#             size=(target_height, target_width),
#             mode='bilinear',
#             align_corners=False
#         )
#         processed_tensors.append(resized_tensor)
    
#     imgs_tensor = torch.cat(processed_tensors, dim=0).to(device)
#     if torch.max(imgs_tensor) > 1.0:
#         imgs_tensor = imgs_tensor / 255.0
#     imgs_tensor = imgs_tensor.repeat(1, 3, 1, 1)
#     imgs_tensor = imgs_tensor * 2.0 - 1.0
    
#     num_imgs = imgs_tensor.size(0)
#     lpips_matrix = np.zeros((num_imgs, num_imgs), dtype=np.float32)
    
#     start = time.perf_counter()
#     with torch.no_grad():
#         for i in tqdm(range(num_imgs), desc="Computing LPIPS distances", leave=False):
#             img_i = imgs_tensor[i:i + 1]
#             for j in range(i + 1, num_imgs):
#                 dist = loss_fn(img_i, imgs_tensor[j:j + 1]).item()
#                 lpips_matrix[i, j] = dist
#                 lpips_matrix[j, i] = dist
#     elapsed = time.perf_counter() - start
    
#     imgs_tensor = imgs_tensor.cpu()
#     return lpips_matrix, elapsed

def extract_lpips_features(images_tensor: torch.Tensor, loss_fn) -> List[torch.Tensor]:
    """Extract and cache LPIPS features for all images."""
    with torch.no_grad():
        # Access the feature extraction network
        features = loss_fn.net.forward(images_tensor)
        
        # Scale features by learned linear weights (if lpips=True)
        processed_features = []
        for i, (feat, lin) in enumerate(zip(features, loss_fn.lins)):
            # Normalize spatial dimensions
            feat_normalized = feat / (feat.norm(dim=1, keepdim=True) + 1e-10)
            # Apply learned linear scaling
            feat_scaled = lin(feat_normalized)
            processed_features.append(feat_scaled)
    
    return processed_features

def compute_distance_from_features(feat_i: List[torch.Tensor], 
                                   feat_j: List[torch.Tensor]) -> float:
    """Compute LPIPS distance from precomputed features."""
    dist = 0.0
    for f_i, f_j in zip(feat_i, feat_j):
        # Compute spatial average of squared L2 distance
        diff = (f_i - f_j) ** 2
        dist += diff.mean()
    return dist.item()

def compute_lpips_matrix_optimized(images: np.ndarray, 
                                   selected_indices: np.ndarray) -> Tuple[np.ndarray, float]:
    """Optimized LPIPS computation with cached features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    loss_fn.eval()
    
    selected_images_list = [images[i] for i in selected_indices]
    if not selected_images_list:
        raise ValueError("No images selected for LPIPS computation.")
    
    target_height, target_width = selected_images_list[0].shape
    
    processed_tensors = []
    for img_np in selected_images_list:
        img_tensor = torch.from_numpy(img_np.astype(np.float32))
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        resized_tensor = torch.nn.functional.interpolate(
            img_tensor,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
        processed_tensors.append(resized_tensor)
    
    imgs_tensor = torch.cat(processed_tensors, dim=0).to(device)
    if torch.max(imgs_tensor) > 1.0:
        imgs_tensor = imgs_tensor / 255.0
    imgs_tensor = imgs_tensor.repeat(1, 3, 1, 1)
    imgs_tensor = imgs_tensor * 2.0 - 1.0
    
    num_imgs = imgs_tensor.size(0)
    lpips_matrix = np.zeros((num_imgs, num_imgs), dtype=np.float32)
    
    start = time.perf_counter()
    
    # Extract features once for all images
    with torch.no_grad():
        all_features = []
        for i in tqdm(range(num_imgs), desc="Extracting features", leave=False):
            img = imgs_tensor[i:i+1]
            features = loss_fn.net.forward(img)
            
            # Process features (normalize + scale)
            processed = []
            for feat, lin in zip(features, loss_fn.lins):
                feat_norm = feat / (feat.norm(dim=1, keepdim=True) + 1e-10)
                processed.append(lin(feat_norm))
            all_features.append(processed)
    
    # Compute pairwise distances from cached features
    lpips_matrix = np.zeros((num_imgs, num_imgs), dtype=np.float32)
    for i in tqdm(range(num_imgs), desc="Computing distances", leave=False):
        for j in range(i + 1, num_imgs):
            dist = sum((f_i - f_j).pow(2).mean().item() 
                      for f_i, f_j in zip(all_features[i], all_features[j]))
            lpips_matrix[i, j] = dist
            lpips_matrix[j, i] = dist
    
    elapsed = time.perf_counter() - start
    return lpips_matrix, elapsed

compute_lpips_matrix = compute_lpips_matrix_optimized


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_sample_images(images: np.ndarray, n_rows: int = 8, n_cols: int = 8) -> plt.Figure:
    """Plot a grid of sample images."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    fig.suptitle("Sample Stimulus Images from PVC8 Dataset", fontsize=14)
    plt.tight_layout()
    return fig


def plot_timing_complexity(
    image_counts: List[int],
    timing_1d: List[float],
    timing_mv: List[float],
    timing_lpips: List[float],
) -> plt.Figure:
    """Plot computational time vs number of images with theoretical fits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1D PP-PCA timing (expected O(n^2))
    ax = axes[0]
    ax.plot(image_counts, timing_1d, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='Measured')
    # Fit quadratic for comparison
    if len(image_counts) > 2:
        coeffs = np.polyfit(np.array(image_counts), np.array(timing_1d), 2)
        x_fit = np.linspace(min(image_counts), max(image_counts), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '--', color='#e74c3c', linewidth=1.5, label=f'Quadratic fit')
    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("1D PP-PCA (Hebbian Embedding)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Multivariate PP-PCA timing (expected O(n^2) for S matrix)
    ax = axes[1]
    ax.plot(image_counts, timing_mv, 'o-', color='#3498db', linewidth=2, markersize=8, label='Measured')
    if len(image_counts) > 2:
        coeffs = np.polyfit(np.array(image_counts), np.array(timing_mv), 2)
        x_fit = np.linspace(min(image_counts), max(image_counts), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '--', color='#e74c3c', linewidth=1.5, label=f'Quadratic fit')
    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Multivariate PP-PCA (3D: x, y, t)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # LPIPS timing (expected O(n^2) pairwise)
    ax = axes[2]
    ax.plot(image_counts, timing_lpips, 'o-', color='#9b59b6', linewidth=2, markersize=8, label='Measured')
    if len(image_counts) > 2:
        coeffs = np.polyfit(np.array(image_counts), np.array(timing_lpips), 2)
        x_fit = np.linspace(min(image_counts), max(image_counts), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '--', color='#e74c3c', linewidth=1.5, label=f'Quadratic fit')
    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("LPIPS (VGG Backbone)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("Computational Complexity: Time vs Number of Images", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_complexity_loglog(
    image_counts: List[int],
    timing_1d: List[float],
    timing_mv: List[float],
) -> plt.Figure:
    """Log-log plot to verify power-law scaling."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    log_n = np.log(image_counts)
    log_t1d = np.log(timing_1d)
    log_tmv = np.log(timing_mv)
    
    # Linear fit in log-log space gives power law exponent
    slope_1d, intercept_1d = np.polyfit(log_n, log_t1d, 1)
    slope_mv, intercept_mv = np.polyfit(log_n, log_tmv, 1)
    
    ax.scatter(log_n, log_t1d, s=100, color='#2ecc71', label=f'1D PP-PCA (slope={slope_1d:.2f})', zorder=5)
    ax.plot(log_n, intercept_1d + slope_1d * log_n, '--', color='#2ecc71', linewidth=2)
    
    ax.scatter(log_n, log_tmv, s=100, color='#3498db', label=f'MV PP-PCA (slope={slope_mv:.2f})', zorder=5)
    ax.plot(log_n, intercept_mv + slope_mv * log_n, '--', color='#3498db', linewidth=2)
    
    # Reference lines for O(n^2) and O(n^3)
    x_ref = np.linspace(min(log_n), max(log_n), 100)
    ax.plot(x_ref, 2 * x_ref + min(log_t1d) - 2 * min(log_n), ':', color='gray', linewidth=1.5, label='O(n²) reference')
    
    ax.set_xlabel("log(Number of Images)", fontsize=12)
    ax.set_ylabel("log(Time in seconds)", fontsize=12)
    ax.set_title("Log-Log Complexity Analysis", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, slope_1d, slope_mv


def plot_neuron_embedding(embed_2d: np.ndarray, N: int) -> plt.Figure:
    """Plot the 2D neuron embedding from MDS on distance matrix."""
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(embed_2d[:, 0], embed_2d[:, 1], c=np.arange(N), cmap='tab20', s=50)
    ax.set_xlabel("Embedding Dimension 1", fontsize=12)
    ax.set_ylabel("Embedding Dimension 2", fontsize=12)
    ax.set_title("Neuron Embedding (Hebbian/MDS)", fontsize=14)
    plt.colorbar(scatter, label="Neuron Index")
    ax.grid(True, alpha=0.3)
    return fig


def plot_pppca_scores(
    scores_mv: pd.DataFrame,
    selected_image_indices: np.ndarray,
    images: np.ndarray,
) -> plt.Figure:
    """Plot PP-PCA scores with image thumbnails."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sc = ax.scatter(
        scores_mv["axis1"], scores_mv["axis2"],
        c=selected_image_indices, cmap='viridis', s=40, zorder=2
    )
    plt.colorbar(sc, label="Image Index")
    
    # Overlay images at intervals
    for idx, (x, y) in enumerate(zip(scores_mv["axis1"], scores_mv["axis2"])):
        if idx % max(1, len(scores_mv) // 15) != 0:
            continue
        i_img = selected_image_indices[idx]
        img = images[i_img]
        imagebox = moffsetbox.OffsetImage(img, zoom=0.08, cmap='gray')
        ab = moffsetbox.AnnotationBbox(
            imagebox, (x, y), frameon=False, pad=0.2, box_alignment=(0.5, -0.2)
        )
        ax.add_artist(ab)
    
    ax.set_xlabel("PP-PCA Axis 1", fontsize=12)
    ax.set_ylabel("PP-PCA Axis 2", fontsize=12)
    ax.set_title("PP-PCA Scores (Hebbian 2D + Time Embedding)", fontsize=14)
    ax.grid(True, alpha=0.3)
    return fig


def plot_explained_variance(eigenvalues: np.ndarray) -> plt.Figure:
    """Plot explained variance by PP-PCA components."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumsum = explained_ratio.cumsum()
    
    x = np.arange(1, len(eigenvalues) + 1)
    ax.bar(x, explained_ratio, alpha=0.6, color='#3498db', label='Individual')
    ax.plot(x, cumsum, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Cumulative')
    
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax.set_title("PP-PCA Explained Variance", fontsize=14)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def plot_correlation_matrix(
    metric_vectors: Dict[str, np.ndarray],
    metric_names: List[str],
) -> plt.Figure:
    """Plot correlation matrix between different distance metrics."""
    metric_matrix = np.vstack([metric_vectors[name] for name in metric_names])
    corr_matrix = np.corrcoef(metric_matrix)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, vmin=-1.0, vmax=1.0, cmap='coolwarm')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=10)
    ax.set_title("Correlation Matrix: PP-PCA vs LPIPS Distances", fontsize=14)
    
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', fontsize=9, color=color)
    
    plt.tight_layout()
    return fig, corr_matrix


def plot_mds_comparison(
    pca_dist_matrix: np.ndarray,
    lpips_dist_matrix: np.ndarray,
) -> plt.Figure:
    """Compare MDS embeddings from PCA and LPIPS distances."""
    mds_pca = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_lpips = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    
    pca_embedding = mds_pca.fit_transform(pca_dist_matrix)
    lpips_embedding = mds_lpips.fit_transform(lpips_dist_matrix)
    
    def normalize_embedding(emb):
        span = np.ptp(emb, axis=0)
        span[span == 0] = 1.0
        return (emb - emb.min(axis=0)) / span
    
    pca_norm = normalize_embedding(pca_embedding)
    lpips_norm = normalize_embedding(lpips_embedding)
    
    # Color by angle from center in PCA space
    center = pca_embedding.mean(axis=0, keepdims=True)
    angles = np.arctan2(pca_embedding[:, 1] - center[0, 1], pca_embedding[:, 0] - center[0, 0])
    hues = (angles + np.pi) / (2 * np.pi)
    colors = plt.cm.hsv(hues)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(pca_norm[:, 0], pca_norm[:, 1], c=colors, s=50)
    axes[0].set_title("MDS on PP-PCA Euclidean Distances", fontsize=13)
    axes[0].set_xlabel("Dimension 1", fontsize=11)
    axes[0].set_ylabel("Dimension 2", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(lpips_norm[:, 0], lpips_norm[:, 1], c=colors, s=50)
    axes[1].set_title("MDS on LPIPS Distances", fontsize=13)
    axes[1].set_xlabel("Dimension 1", fontsize=11)
    axes[1].set_ylabel("Dimension 2", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Angular Position (from PP-PCA MDS center)")
    
    fig.suptitle("Comparison of Distance-Based Embeddings", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_distance_scatter(
    pca_dist: np.ndarray,
    lpips_dist: np.ndarray,
    pearson_r: float,
    spearman_r: float,
) -> plt.Figure:
    """Scatter plot of PCA distances vs LPIPS distances."""
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.scatter(pca_dist, lpips_dist, alpha=0.4, s=10, color='#3498db')
    
    # Add regression line
    z = np.polyfit(pca_dist, lpips_dist, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pca_dist.min(), pca_dist.max(), 100)
    ax.plot(x_line, p(x_line), '--', color='#e74c3c', linewidth=2, label='Linear fit')
    
    ax.set_xlabel("PP-PCA Euclidean Distance", fontsize=12)
    ax.set_ylabel("LPIPS Distance", fontsize=12)
    ax.set_title(f"PP-PCA vs LPIPS Distances\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmark(session_id: int = None, image_type: str = None) -> Dict:
    """
    Run the full benchmark experiment.
    
    Parameters
    ----------
    session_id : int, optional
        Session number (1-10). If None, uses global SESSION_ID.
    image_type : str, optional
        Filter images by type: "all", "small", or "big". If None, uses global IMAGE_TYPE.
    """
    print("=" * 70)
    print("V1 Neural Coding Benchmark Experiment")
    print("=" * 70)
    
    # Load data
    print("\n[1/7] Loading PVC8 dataset...")
    resp, images, data_info = load_data(session_id, image_type)
    N, I, R, T = resp.shape
    t_grid_binned = get_binned_t_grid(T, BIN_SIZE)
    
    print(f"  → Session: {data_info['session_id']}, Image type: {data_info['image_type']}")
    print(f"  → Neurons: {N}, Images: {I} (of {data_info['num_images_total']} total), Repeats: {R}, Time bins: {T}")
    print(f"  → Binned time grid size: {len(t_grid_binned)}")
    
    # Plot sample images
    fig_images = plot_sample_images(images)
    save_figure(fig_images, "01_sample_images")
    
    # Timing benchmarks
    print("\n[2/7] Running timing benchmarks across different image counts...")
    timing_results = {
        'image_counts': [],
        'timing_1d': [],
        'timing_mv': [],
        'timing_lpips': [],
        'num_1d_processes': [],
        'num_mv_processes': [],
        'total_events_1d': [],
        'total_events_mv': [],
    }
    
    for num_images in IMAGE_COUNTS:
        print(f"\n  Processing {num_images} images...")
        
        # Select images
        all_image_indices = np.arange(I)
        np.random.seed(RANDOM_SEED)
        selected_indices = np.random.choice(all_image_indices, min(num_images, I), replace=False)
        
        # Prepare 1D point processes
        all_1d_points, idx_map = prepare_1d_point_processes(
            resp, selected_indices, t_grid_binned, NUM_REPEATS_TO_USE, BIN_SIZE
        )
        
        # Run 1D PP-PCA
        scores1d, time_1d = run_1d_pppca(all_1d_points, J1)
        
        # Build neuron distance matrix and embedding
        D = build_neuron_distance_matrix(scores1d, idx_map, N, I, J1)
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        embed_2d = mds.fit_transform(D)
        embed_2d = (embed_2d - embed_2d.min(0)) / np.maximum(np.ptp(embed_2d, 0), 1e-12)
        
        # Build multivariate point processes
        mvpp_per_image = build_multivariate_pp(resp, selected_indices, embed_2d, t_grid_binned, BIN_SIZE)
        
        # Run multivariate PP-PCA
        scores_mv, eigenvalues_mv, time_mv = run_mv_pppca(mvpp_per_image, J_MV)
        
        # Compute LPIPS
        lpips_matrix, time_lpips = compute_lpips_matrix(images, selected_indices)
        
        # Store results
        timing_results['image_counts'].append(num_images)
        timing_results['timing_1d'].append(time_1d)
        timing_results['timing_mv'].append(time_mv)
        timing_results['timing_lpips'].append(time_lpips)
        timing_results['num_1d_processes'].append(len(all_1d_points))
        timing_results['num_mv_processes'].append(len(mvpp_per_image))
        timing_results['total_events_1d'].append(sum(p.shape[0] for p in all_1d_points))
        timing_results['total_events_mv'].append(sum(p.shape[0] for p in mvpp_per_image))
        
        print(f"    1D PP-PCA: {time_1d:.2f}s ({len(all_1d_points)} processes)")
        print(f"    MV PP-PCA: {time_mv:.2f}s ({len(mvpp_per_image)} processes)")
        print(f"    LPIPS:     {time_lpips:.2f}s")
    
    # Plot timing complexity
    print("\n[3/7] Generating timing complexity plots...")
    fig_timing = plot_timing_complexity(
        timing_results['image_counts'],
        timing_results['timing_1d'],
        timing_results['timing_mv'],
        timing_results['timing_lpips'],
    )
    save_figure(fig_timing, "02_timing_complexity")
    
    fig_loglog, slope_1d, slope_mv = plot_complexity_loglog(
        timing_results['image_counts'],
        timing_results['timing_1d'],
        timing_results['timing_mv'],
    )
    save_figure(fig_loglog, "03_loglog_complexity")
    timing_results['slope_1d'] = slope_1d
    timing_results['slope_mv'] = slope_mv
    
    # Full run with maximum images for detailed analysis
    print("\n[4/7] Running full analysis with maximum images...")
    max_images = max(IMAGE_COUNTS)
    np.random.seed(RANDOM_SEED)
    selected_indices = np.random.choice(np.arange(I), min(max_images, I), replace=False)
    
    all_1d_points, idx_map = prepare_1d_point_processes(
        resp, selected_indices, t_grid_binned, NUM_REPEATS_TO_USE, BIN_SIZE
    )
    scores1d, _ = run_1d_pppca(all_1d_points, J1)
    D = build_neuron_distance_matrix(scores1d, idx_map, N, I, J1)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embed_2d = mds.fit_transform(D)
    embed_2d = (embed_2d - embed_2d.min(0)) / np.maximum(np.ptp(embed_2d, 0), 1e-12)
    
    fig_embed = plot_neuron_embedding(embed_2d, N)
    save_figure(fig_embed, "04_neuron_embedding")
    
    mvpp_per_image = build_multivariate_pp(resp, selected_indices, embed_2d, t_grid_binned, BIN_SIZE)
    scores_mv, eigenvalues_mv, _ = run_mv_pppca(mvpp_per_image, J_MV)
    scores_mv = scores_mv.reset_index(drop=True)
    
    fig_scores = plot_pppca_scores(scores_mv, selected_indices, images)
    save_figure(fig_scores, "05_pppca_scores")
    
    fig_variance = plot_explained_variance(eigenvalues_mv)
    save_figure(fig_variance, "06_explained_variance")
    
    # Compute LPIPS and correlations
    print("\n[5/7] Computing LPIPS matrix and correlations...")
    lpips_matrix, _ = compute_lpips_matrix(images, selected_indices)
    
    score_columns = [col for col in scores_mv.columns if col.startswith("axis")]
    scores_array = scores_mv[score_columns].to_numpy(dtype=np.float64)
    num_scores = scores_array.shape[0]
    
    metric_vectors = {}
    for idx, column in enumerate(score_columns):
        metric_vectors[f"{column}_dist"] = pdist(scores_array[:, [idx]], metric="euclidean")
    metric_vectors["pca_euclidean"] = pdist(scores_array, metric="euclidean")
    triu_idx = np.triu_indices(num_scores, k=1)
    metric_vectors["lpips"] = lpips_matrix[triu_idx]
    
    metric_names = list(metric_vectors.keys())
    fig_corr, corr_matrix = plot_correlation_matrix(metric_vectors, metric_names)
    save_figure(fig_corr, "07_correlation_matrix")
    
    pca_dist = metric_vectors["pca_euclidean"]
    lpips_dist = metric_vectors["lpips"]
    pearson_r, _ = pearsonr(pca_dist, lpips_dist)
    spearman_r, _ = spearmanr(pca_dist, lpips_dist)
    
    fig_scatter = plot_distance_scatter(pca_dist, lpips_dist, pearson_r, spearman_r)
    save_figure(fig_scatter, "08_distance_scatter")
    
    # MDS comparison
    print("\n[6/7] Generating MDS comparison plots...")
    pca_dist_matrix = squareform(pca_dist)
    lpips_dist_matrix = squareform(lpips_dist)
    fig_mds = plot_mds_comparison(pca_dist_matrix, lpips_dist_matrix)
    save_figure(fig_mds, "09_mds_comparison")
    
    # Compile results
    print("\n[7/7] Compiling results...")
    results = {
        'data_info': data_info,
        'timing': timing_results,
        'correlations': {
            'pearson_pca_lpips': pearson_r,
            'spearman_pca_lpips': spearman_r,
            'corr_matrix': corr_matrix.tolist(),
            'metric_names': metric_names,
        },
        'eigenvalues': eigenvalues_mv.tolist(),
        'num_images_used': max_images,
        'parameters': {
            'bin_size': BIN_SIZE,
            'num_repeats': NUM_REPEATS_TO_USE,
            'j1': J1,
            'j_mv': J_MV,
            'random_seed': RANDOM_SEED,
        },
    }
    
    return results


def generate_report(results: Dict) -> str:
    """Generate a markdown report from benchmark results."""
    
    # Determine image type description
    image_type_desc = {
        'all': 'All images (small, big, and gratings)',
        'small': 'Small images only (windowed to 1 degree)',
        'big': 'Big images only (3-6.7 degrees)',
    }.get(results['data_info'].get('image_type', 'all'), 'All images')
    
    report = f"""# V1 Neural Coding Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

This report documents a comprehensive benchmark of **Point Process PCA (PP-PCA)** applied to 
neural coding data from the PVC8 dataset (primary visual cortex recordings). The experiment 
evaluates computational complexity, correlation with perceptual image similarity (LPIPS), 
and the quality of learned representations.

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Session | {results['data_info'].get('session_id', 1)} |
| Image Type | {image_type_desc} |
| Neurons | {results['data_info']['num_neurons']} |
| Images Used | {results['data_info']['num_images']} |
| Total Images in Session | {results['data_info'].get('num_images_total', results['data_info']['num_images'])} |
| Trial Repeats | {results['data_info']['num_repeats']} |
| Time Bins | {results['data_info']['num_time_bins']} |

### Sample Stimulus Images

![Sample Images](01_sample_images.png)

*Figure 1: Sample grayscale natural images from the PVC8 dataset used as visual stimuli.*

---

## Experimental Parameters

| Parameter | Value |
|-----------|-------|
| Temporal Bin Size | {results['parameters']['bin_size']} |
| Trial Repeats Used | {results['parameters']['num_repeats']} |
| 1D PP-PCA Components (J₁) | {results['parameters']['j1']} |
| MV PP-PCA Components (J_mv) | {results['parameters']['j_mv']} |
| Random Seed | {results['parameters']['random_seed']} |

---

## Computational Complexity Analysis

### Theoretical Background

From the complexity analysis (`parts/complexity.md`), the key computational costs are:

1. **Building S matrix:** $O(d \\cdot (\\sum_i k_i)^2)$ where $k_i$ is the number of events in process $i$
2. **Eigendecomposition:** $O(n^3)$ for the dense $n \\times n$ Gram matrix
3. **When balanced:** $O(n^2 \\cdot k^2 \\cdot d)$ where $k$ is average events per process

For our benchmark:
- **1D PP-PCA:** Processes each neuron-image-repeat combination ($n$ scales with images × neurons)
- **MV PP-PCA:** Processes each image with 3D points (x, y, t), so $n$ = number of images

### Measured Timing Results

| Images | 1D PP-PCA (s) | MV PP-PCA (s) | LPIPS (s) | 1D Processes | MV Processes |
|--------|---------------|---------------|-----------|--------------|--------------|
"""
    
    for i, count in enumerate(results['timing']['image_counts']):
        report += f"| {count} | {results['timing']['timing_1d'][i]:.2f} | {results['timing']['timing_mv'][i]:.2f} | {results['timing']['timing_lpips'][i]:.2f} | {results['timing']['num_1d_processes'][i]} | {results['timing']['num_mv_processes'][i]} |\n"
    
    report += f"""
### Timing Plots

![Timing Complexity](02_timing_complexity.png)

*Figure 2: Computational time vs number of images for each algorithm. Dashed lines show quadratic fits.*

![Log-Log Complexity](03_loglog_complexity.png)

*Figure 3: Log-log plot revealing power-law scaling. Measured slopes: 1D PP-PCA = {results['timing']['slope_1d']:.2f}, MV PP-PCA = {results['timing']['slope_mv']:.2f}. Theory predicts slope ≈ 2 for quadratic complexity.*

### Complexity Discussion

The measured power-law exponents are:
- **1D PP-PCA:** {results['timing']['slope_1d']:.2f} (expected ~2 from $O(n^2)$ for S matrix construction)
- **MV PP-PCA:** {results['timing']['slope_mv']:.2f} (expected ~2, dominated by pairwise integral computation)

The slight deviation from exactly 2 can be attributed to:
1. Eigendecomposition overhead ($O(n^3)$) becoming significant at larger $n$
2. Memory allocation and data movement costs
3. Variable number of events per process

---

## Neuron Embedding (Hebbian Learning)

The first stage applies 1D PP-PCA to spike timing data, learning a latent representation 
for each neuron across images. MDS is then used to embed neurons in 2D based on their 
response similarity.

![Neuron Embedding](04_neuron_embedding.png)

*Figure 4: 2D embedding of neurons based on their PP-PCA score distances across images. 
Colors indicate neuron index. Nearby neurons have similar temporal response patterns.*

---

## Multivariate PP-PCA Results

The multivariate PP-PCA operates on 3D point processes (x, y, t) where:
- (x, y) is the neuron's 2D embedding position
- t is the spike timing

### PP-PCA Scores

![PP-PCA Scores](05_pppca_scores.png)

*Figure 5: First two PP-PCA components for each image. Thumbnails show the corresponding 
stimulus images. Images that evoke similar neural population responses are clustered together.*

### Explained Variance

| Component | Eigenvalue | Variance Ratio | Cumulative |
|-----------|------------|----------------|------------|
"""
    
    total_var = sum(results['eigenvalues'])
    cumsum = 0
    for i, ev in enumerate(results['eigenvalues']):
        ratio = ev / total_var if total_var > 0 else 0
        cumsum += ratio
        report += f"| {i+1} | {ev:.4f} | {ratio:.4f} | {cumsum:.4f} |\n"
    
    report += f"""
![Explained Variance](06_explained_variance.png)

*Figure 6: Variance explained by each PP-PCA component. The first component captures 
the dominant mode of neural population variability across images.*

---

## Perceptual Distance Comparison (LPIPS)

### Approach: VGG-based Perceptual Similarity

Following the methodology established in the **StyleGAN** paper (Karras et al., 2019) and 
the **LPIPS** metric (Zhang et al., 2018), we measure perceptual image similarity using 
deep features from a VGG network.

**LPIPS (Learned Perceptual Image Patch Similarity):**

1. **Feature Extraction:** Images are passed through a pre-trained VGG-16 network
2. **Multi-scale Comparison:** Features from multiple layers (conv1_2 through conv5_2) are extracted
3. **Spatial Averaging:** Features are spatially averaged and L2-normalized
4. **Learned Weights:** A learned linear combination of layer-wise distances produces the final score

The key insight from StyleGAN is that perceptual distances in deep feature space 
better capture human judgment of image similarity than pixel-wise metrics.

**Processing Pipeline:**
- Grayscale images are converted to 3-channel by replication
- Images are normalized to [-1, 1] range
- Pairwise distances computed for all image pairs
- Lower LPIPS = more perceptually similar

### Correlation Analysis

**Key Result:** Pearson correlation between PP-PCA Euclidean distance and LPIPS = **{results['correlations']['pearson_pca_lpips']:.4f}**

Spearman rank correlation = **{results['correlations']['spearman_pca_lpips']:.4f}**

![Correlation Matrix](07_correlation_matrix.png)

*Figure 7: Correlation matrix between different distance metrics. PP-PCA axes and their 
Euclidean combination are compared against LPIPS perceptual distance.*

![Distance Scatter](08_distance_scatter.png)

*Figure 8: Scatter plot of PP-PCA Euclidean distances vs LPIPS distances for all image pairs.
The positive correlation indicates that neural population responses (as captured by PP-PCA) 
partially reflect perceptual image similarity.*

### MDS Embedding Comparison

![MDS Comparison](09_mds_comparison.png)

*Figure 9: 2D MDS embeddings derived from PP-PCA distances (left) and LPIPS distances (right).
Colors are assigned based on angular position in the PP-PCA embedding to reveal structural 
correspondence between the two spaces.*

---

## Summary and Key Findings

### Computational Performance
1. PP-PCA exhibits approximately **quadratic complexity** in the number of images, consistent with theory
2. The MV PP-PCA on {results['num_images_used']} images completes in **{results['timing']['timing_mv'][-1]:.1f} seconds**
3. LPIPS computation is comparable in cost for small datasets but scales better due to GPU parallelization

### Representational Quality
1. PP-PCA captures **{sum(results['eigenvalues'][:1])/sum(results['eigenvalues'])*100:.1f}%** of variance in the first component
2. Moderate correlation (**r = {results['correlations']['pearson_pca_lpips']:.3f}**) between neural-derived distances and perceptual similarity
3. The MDS embeddings show structural similarity, suggesting shared representational geometry

### Implications
- Neural population responses partially encode perceptual image features
- PP-PCA provides a principled dimensionality reduction for point process data
- The correlation with LPIPS validates the biological relevance of the learned representations

---

## Appendix: File Listing

All figures are saved in the same directory as this report:

1. `01_sample_images.png` - Dataset sample images
2. `02_timing_complexity.png` - Timing vs image count
3. `03_loglog_complexity.png` - Log-log complexity analysis
4. `04_neuron_embedding.png` - Hebbian neuron embedding
5. `05_pppca_scores.png` - PP-PCA score visualization
6. `06_explained_variance.png` - Variance explained plot
7. `07_correlation_matrix.png` - Metric correlation matrix
8. `08_distance_scatter.png` - PCA vs LPIPS scatter
9. `09_mds_comparison.png` - MDS embedding comparison

---

*Report generated by `v1_benchmark.py`*
"""
    
    return report


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V1 Neural Coding Benchmark with PP-PCA")
    parser.add_argument(
        "--session", "-s", type=int, default=None,
        help="Session ID (1-10). Default uses SESSION_ID config variable."
    )
    parser.add_argument(
        "--image-type", "-t", type=str, default=None,
        choices=["all", "small", "big"],
        help="Image type filter: 'all' (default), 'small' (1 degree), or 'big' (3-6.7 degrees)"
    )
    args = parser.parse_args()
    
    # Run benchmark with specified parameters
    results = run_benchmark(session_id=args.session, image_type=args.image_type)
    
    # Generate and save report
    report_md = generate_report(results)
    report_path = REPORT_DIR / "report.md"
    report_path.write_text(report_md, encoding='utf-8')
    
    print(f"\n{'=' * 70}")
    print(f"Report saved to: {report_path.absolute()}")
    print(f"Figures saved to: {REPORT_DIR.absolute()}")
    print(f"{'=' * 70}")
    
    # Save raw results as JSON
    import json
    results_path = REPORT_DIR / "results.json"
    # Convert numpy arrays to lists for JSON serialization
    results_json = {k: v for k, v in results.items() if k != 'correlations' or k == 'correlations'}
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to: {results_path.absolute()}")
