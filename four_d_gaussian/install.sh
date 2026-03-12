#!/bin/bash
# Master Installation Script for 4D Gaussian Environment
set -e

echo "============================================================"
echo "🚀 Starting Master Installation"
echo "============================================================"

# 1. System Dependencies
echo "📦 [1/8] Installing system dependencies..."
apt-get update && apt-get install -y \
    colmap ffmpeg xvfb libgl1-mesa-glx python3-opencv sqlite3 \
    curl libx11-dev tree wget git-lfs libcudnn9-dev-cuda-12

git lfs install

# 2. Environment Setup
echo "🌐 [2/8] Setting up environment variables..."
export CUDA_HOME=/usr/local/cuda-12.4
export CUDNN_PATH=/usr/local/cuda
export TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.11/dist-packages/nvidia/curand/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Crucial for finding the correct PyTorch headers when multiple versions are present
export CPATH=$TORCH_PATH/include:$TORCH_PATH/include/torch/csrc/api/include:$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CPATH:$CPLUS_INCLUDE_PATH

export NVTE_CUDA_INCLUDE_PATH=$CUDA_HOME/include
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Clear caches to avoid ABI pollution
rm -rf ~/.cache/torch_extensions

# 3. Python Requirements
echo "🐍 [3/8] Installing Python requirements..."
pip install --upgrade pip
pip install hf_transfer decord tyro nltk better_profanity boto3 accelerate tqdm
pip install "imageio[ffmpeg,pyav]>=2.37.0"
pip install scipy pandas retina-face megatron-core scikit-image matplotlib "pydantic[email]"
pip install nvidia-ml-py --upgrade
pip install ninja
export MAX_JOBS=1 && pip install --no-cache-dir --no-build-isolation --no-binary flash-attn flash-attn==2.7.3 --verbose
pip install --no-build-isolation "transformer-engine[pytorch]>=2.12.0"
pip install natten
pip install pycolmap

ROOT_DIR="/workspace/sim-animate-environment"

if [ -f "$ROOT_DIR/infinite-simul-realtime-4d-gaussian-vgg/requirements.txt" ]; then
    pip install -r $ROOT_DIR/infinite-simul-realtime-4d-gaussian-vgg/requirements.txt
fi
if [ -f "$ROOT_DIR/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite_simul_spacetime_gaussian/requirements.txt" ]; then
    pip install -r $ROOT_DIR/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite_simul_spacetime_gaussian/requirements.txt
fi

pip install easydict albumentations better-profanity boto3 click diffusers einops ftfy fvcore fastparquet huggingface-hub hydra-core "imageio[pyav,ffmpeg]" iopath loguru mediapy ml-dtypes modelscope multi-storage-client numpydantic omegaconf opencv-python pandas pyarrow peft pydantic qwen-vl-utils retinaface-py timm transformers wandb webdataset xformers angelslim==0.2.2 runwayml

# 4. Cosmos Predict 2.5 Installation
echo "🌌 [4/8] Installing Cosmos Predict 2.5..."
if [ -d "$ROOT_DIR/cosmos-predict2.5" ]; then
    cd $ROOT_DIR/cosmos-predict2.5
    git lfs pull
    pip install -e packages/cosmos-oss
    pip install -e packages/cosmos-cuda
    pip install -e packages/cosmos-gradio
    pip install -e .
    cd $ROOT_DIR
fi

# 5. CUDA Submodule Compilation (Gaussian Splatting)
echo "⚙️ [5/8] Compiling custom CUDA modules..."
GAUSSIAN_SUBMODULES="$ROOT_DIR/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite_simul_spacetime_gaussian/thirdparty/gaussian_splatting/submodules"
modules=("forward_full" "forward_lite" "gaussian_rasterization_ch3" "gaussian_rasterization_ch9" "simple-knn")

for module in "${modules[@]}"; do
    if [ -d "$GAUSSIAN_SUBMODULES/$module" ]; then
        echo "   -> Compiling $module..."
        cd "$GAUSSIAN_SUBMODULES/$module"
        pip install . --no-build-isolation
    fi
done

# 6. MMCV from source
echo "🔨 [6/8] Compiling MMCV from source..."
MMCV_PATH="$ROOT_DIR/infinite-simul-realtime-4d-gaussian-vgg/third_party/infinite_simul_spacetime_gaussian/thirdparty/mmcv"
if [ -d "$MMCV_PATH" ]; then
    cd "$MMCV_PATH"
    # Clean previous failed builds
    python3 setup.py clean --all || true
    rm -rf build mmcv.egg-info
    find . -name "*.so" -delete
    
    pip install addict mmengine yapf
    MMCV_WITH_OPS=1 pip install . --no-build-isolation --verbose
fi

# 7. HF CLI Tool
if [ ! -f "/usr/local/bin/hf" ]; then
    echo "⬇️ [7/8] Installing optimized HF CLI..."
    wget -O /tmp/hf https://github.com/michaelfdf/hf/releases/latest/download/hf-linux-amd64 || curl -L -o /tmp/hf https://github.com/michaelfdf/hf/releases/latest/download/hf-linux-amd64
    mv /tmp/hf /usr/local/bin/hf
    chmod +x /usr/local/bin/hf
fi

# 8. Sharp Monocular View Synthesis
echo "📸 [8/8] Installing Sharp Monocular View Synthesis..."
if [ -d "$ROOT_DIR/ml-sharp" ]; then
    cd "$ROOT_DIR/ml-sharp"
    # Note: Using pip install -e . as per requirements.txt -e . entry
    # This installs the project in editable mode along with its dependencies
    pip install -e .
    
    # Download pre-trained model checkpoint
    if [ ! -f "sharp_2572gikvuh.pt" ]; then
        echo "   -> Downloading Sharp model checkpoint..."
        wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
    fi
    cd "$ROOT_DIR"
fi

echo "============================================================"
echo "✅ Installation Complete!"
echo "============================================================"
