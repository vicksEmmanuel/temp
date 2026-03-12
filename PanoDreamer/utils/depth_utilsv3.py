"""
Depth estimation using Depth Anything 3.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

# Add Depth-Anything-3/src to path for internal imports if not installed
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DA3_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), 'Depth-Anything-3', 'src')
if _DA3_PATH not in sys.path:
    sys.path.insert(0, _DA3_PATH)

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("[WARNING] Depth-Anything-3 package not found in path. Ensure it's installed or DA3_PATH is correct.")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Lazy loading - model initialized on first use
_model = None
# Default to DA3NESTED-GIANT-LARGE as suggested in the docs for best results
_model_repo = "depth-anything/DA3NESTED-GIANT-LARGE"
_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), 'checkpoints')

def _get_model():
    """Load model on first use."""
    global _model
    if _model is None:
        print(f'[INFO] Loading Depth Anything 3 from {_model_repo}')
        print(f'[INFO] Using checkpoints directory: {_CHECKPOINTS_DIR}')
        # We assume from_pretrained supports cache_dir or we set it via env var if needed
        # Most HF-compatible APIs support cache_dir
        _model = DepthAnything3.from_pretrained(
            _model_repo, 
            cache_dir=_CHECKPOINTS_DIR
        )
        _model = _model.to(device=DEVICE)
    return _model

def estimate_depth(img, mode='test'):
    """
    Estimate depth from an image using Depth Anything 3.
    
    Args:
        img: PIL Image or numpy array (RGB)
        mode: Unused, kept for compatibility
        
    Returns:
        depth: HxW numpy array of depth values
    """
    model = _get_model()
    
    # DA3 model expects a list of images (or a single image in a list)
    # The API expects PIL images or paths
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img

    with torch.no_grad():
        # inference returns a prediction object
        prediction = model.inference([img_pil])
        
    # prediction.depth is [N, H, W], we want [H, W]
    depth = prediction.depth[0]

    return depth
