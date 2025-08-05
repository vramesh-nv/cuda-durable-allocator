#!/bin/bash

# Installation script for GPU Memory FUSE dependencies
# Supports Ubuntu/Debian and CentOS/RHEL/Fedora

set -e

echo "GPU Memory FUSE Dependency Installer"
echo "====================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "Cannot detect OS. Please install dependencies manually."
    exit 1
fi

echo "Detected OS: $OS"

# Function to install on Ubuntu/Debian
install_ubuntu_debian() {
    echo "Installing dependencies for Ubuntu/Debian..."
    
    sudo apt update
    
    # Basic build tools
    sudo apt install -y build-essential cmake pkg-config
    
    # FUSE development
    sudo apt install -y libfuse3-dev fuse3
    
    # GLib development
    sudo apt install -y libglib2.0-dev
    
    # Development tools
    sudo apt install -y clang-format attr
    
    echo "Basic dependencies installed successfully!"
    echo ""
    echo "CUDA Toolkit Installation:"
    echo "Please install CUDA Toolkit manually from:"
    echo "https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "Or for Ubuntu, you can try:"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"
    echo "  sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb"
    echo "  sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb"
    echo "  sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/"
    echo "  sudo apt update"
    echo "  sudo apt install -y cuda"
}

# Function to install on CentOS/RHEL/Fedora
install_redhat() {
    echo "Installing dependencies for CentOS/RHEL/Fedora..."
    
    # Detect package manager
    if command -v dnf &> /dev/null; then
        PKG_MGR="dnf"
    elif command -v yum &> /dev/null; then
        PKG_MGR="yum"
    else
        echo "No supported package manager found (dnf/yum)"
        exit 1
    fi
    
    # Basic build tools
    sudo $PKG_MGR install -y gcc gcc-c++ make cmake pkgconfig
    
    # FUSE development
    sudo $PKG_MGR install -y fuse3-devel fuse3
    
    # GLib development
    sudo $PKG_MGR install -y glib2-devel
    
    # Development tools
    sudo $PKG_MGR install -y clang-tools-extra attr
    
    echo "Basic dependencies installed successfully!"
    echo ""
    echo "CUDA Toolkit Installation:"
    echo "Please install CUDA Toolkit manually from:"
    echo "https://developer.nvidia.com/cuda-downloads"
}

# Main installation logic
case "$OS" in
    "Ubuntu"|"Debian GNU/Linux")
        install_ubuntu_debian
        ;;
    "CentOS Linux"|"Red Hat Enterprise Linux"|"Fedora")
        install_redhat
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Please install the following dependencies manually:"
        echo "  - build-essential/gcc-c++"
        echo "  - libfuse3-dev/fuse3-devel"
        echo "  - libglib2.0-dev/glib2-devel"
        echo "  - CUDA Toolkit"
        exit 1
        ;;
esac

echo ""
echo "Post-installation setup:"
echo "1. Add CUDA to your PATH:"
echo "   export PATH=/usr/local/cuda/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "2. Add to your shell profile (~/.bashrc or ~/.zshrc):"
echo "   echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc"
echo "   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo ""
echo "3. Ensure your user can access CUDA devices:"
echo "   sudo usermod -a -G video \$USER"
echo "   # Log out and log back in for group changes to take effect"
echo ""
echo "4. Load FUSE module if not already loaded:"
echo "   sudo modprobe fuse"
echo ""
echo "5. Test CUDA installation:"
echo "   nvcc --version"
echo "   nvidia-smi"
echo ""
echo "6. Verify dependencies:"
echo "   make check-deps"
echo ""
echo "Installation script completed!"

# Check if CUDA is already available
if command -v nvcc &> /dev/null; then
    echo ""
    echo "CUDA appears to be already installed:"
    nvcc --version
else
    echo ""
    echo "CUDA not found in PATH. Please install CUDA Toolkit."
fi