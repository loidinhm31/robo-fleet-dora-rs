#!/bin/bash
# QuickStart script for PyBullet Simulation
# Automates setup and testing

set -e  # Exit on error

echo "================================================"
echo "PyBullet Simulation - QuickStart Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python ${PYTHON_VERSION} found"

# Check if running from correct directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "✓ Working directory: $SCRIPT_DIR"

# Find repo root and URDF
REPO_ROOT="$(cd ../.. && pwd)"
URDF_PATH="${REPO_ROOT}/model/LeKiwi.urdf"

if [ ! -f "$URDF_PATH" ]; then
    echo -e "${RED}Error: LeKiwi.urdf not found at $URDF_PATH${NC}"
    echo "Please ensure the URDF file exists"
    exit 1
fi
echo "✓ Found URDF: $URDF_PATH"

# Check for virtual environment (create if needed)
echo -e "\n${YELLOW}[2/6] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Use venv python for the rest of the script
PYTHON="$SCRIPT_DIR/venv/bin/python3"
PIP="$SCRIPT_DIR/venv/bin/pip"

# Install dependencies
echo -e "\n${YELLOW}[3/6] Installing dependencies...${NC}"
$PIP install -q --upgrade pip
$PIP install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check for dora-rs
echo -e "\n${YELLOW}[4/6] Checking dora-rs...${NC}"
if $PYTHON -c "import dora" 2>/dev/null; then
    echo "✓ dora-rs Python bindings installed"
else
    echo -e "${YELLOW}Note: dora-rs not installed (optional for standalone testing)${NC}"
    echo "Install with: $PIP install dora-rs"
fi

# Set environment variable
export URDF_PATH="$URDF_PATH"
export GUI_ENABLED="true"

# Test configuration loading
echo -e "\n${YELLOW}[5/6] Testing configuration...${NC}"
if $PYTHON pybullet_config.py > /dev/null 2>&1; then
    echo "✓ Configuration validated"
else
    echo -e "${RED}Error: Configuration test failed${NC}"
    echo "Running diagnostics..."
    $PYTHON pybullet_config.py
    exit 1
fi

# Run quick joint mapping test
echo -e "\n${YELLOW}[6/6] Testing joint mapping...${NC}"
if $PYTHON test_joints.py 2>&1 | grep -q "SUCCESS"; then
    echo "✓ Joint mapping verified: 3 wheel joints + 6 arm joints"
else
    echo -e "${RED}Error: Joint mapping failed${NC}"
    $PYTHON test_joints.py
    exit 1
fi

echo -e "\n${GREEN}✓ All tests passed!${NC}"
echo ""
echo "To run full tests with GUI:"
echo "  $PYTHON test_standalone.py"
echo ""
echo "To integrate with dora-rs:"
echo "  See INTEGRATION.md for dataflow configuration"

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}QuickStart Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
