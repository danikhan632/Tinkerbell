#!/bin/bash
# Script to install transformer_engine with all dependencies from source
# This script handles all the complex dependencies and build configuration

set -e  # Exit on error

echo "==================================================================="
echo "Transformer Engine Installation Script"
echo "==================================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root for apt operations
if [ "$EUID" -ne 0 ]; then
    print_warning "Some operations require sudo privileges"
    SUDO="sudo"
else
    SUDO=""
fi

# Step 1: Install system dependencies
print_status "Installing system dependencies..."
$SUDO apt update
$SUDO apt install -y \
    build-essential \
    cmake \
    ninja-build \
    g++-12 \
    gcc-12 \
    python3.12-dev \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    libnccl-dev \
    libnccl2

# Step 2: Set up CUDA environment
print_status "Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda-12
export CUDA_PATH=/usr/local/cuda-12
export CUDNN_PATH=/usr
export PATH="/usr/local/cuda-12/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
export MAX_JOBS=4
export NVTE_FRAMEWORK=pytorch

# Step 3: Create symlinks for cuDNN headers (if needed)
print_status "Creating cuDNN symlinks..."
if [ ! -f "/usr/local/cuda-12/include/cudnn.h" ]; then
    $SUDO ln -sf /usr/include/x86_64-linux-gnu/cudnn*.h /usr/local/cuda-12/include/
    $SUDO ln -sf /usr/lib/x86_64-linux-gnu/libcudnn*.so* /usr/local/cuda-12/lib64/
    print_status "cuDNN symlinks created"
else
    print_status "cuDNN symlinks already exist"
fi

# Step 4: Install Python dependencies
print_status "Installing Python dependencies..."
/usr/bin/python3.12 -m pip install --user \
    pybind11 \
    nvidia-mathdx \
    torch \
    numpy

# Step 5: Clone or update transformer_engine repository
REPO_DIR="/tmp/TransformerEngine"
if [ -d "$REPO_DIR" ]; then
    print_status "Transformer Engine repository already exists at $REPO_DIR"
    cd "$REPO_DIR"
    print_status "Pulling latest changes..."
    git pull || true
else
    print_status "Cloning Transformer Engine repository..."
    cd /tmp
    git clone https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
fi

# Step 6: Checkout specific version and initialize submodules
print_status "Checking out v2.8 and initializing submodules..."
git checkout v2.8
git submodule update --init --recursive

# Step 7: Clean previous builds
print_status "Cleaning previous builds..."
rm -rf build dist *.egg-info

# Step 8: Build transformer_engine wheel
print_status "Building transformer_engine wheel (this will take several minutes)..."
/usr/bin/python3.12 setup.py bdist_wheel

# Step 9: Install the built wheel
print_status "Installing transformer_engine..."
WHEEL_FILE=$(ls dist/transformer_engine*.whl | head -n 1)
if [ -f "$WHEEL_FILE" ]; then
    /usr/bin/python3.12 -m pip install --user --force-reinstall "$WHEEL_FILE"
    print_status "✓ Transformer Engine installed successfully!"
    print_status "Wheel location: $WHEEL_FILE"
else
    print_error "Build failed - wheel file not found"
    exit 1
fi

# Step 10: Verify installation
print_status "Verifying installation..."
/usr/bin/python3.12 -c "import transformer_engine.pytorch as te; print('Transformer Engine version:', te.__version__)" && \
    print_status "✓ Installation verified successfully!" || \
    print_error "Installation verification failed"

echo ""
echo "==================================================================="
echo "Installation Complete!"
echo "==================================================================="
echo ""
echo "To use transformer_engine in your code, make sure to set these"
echo "environment variables in your shell:"
echo ""
echo "export CUDA_HOME=/usr/local/cuda-12"
echo "export PATH=\"/usr/local/cuda-12/bin:\$PATH\""
echo "export LD_LIBRARY_PATH=\"/usr/local/cuda-12/lib64:\$LD_LIBRARY_PATH\""
echo ""
echo "Example usage:"
echo "  import transformer_engine.pytorch as te"
echo "  from transformer_engine.common import recipe"
echo ""
