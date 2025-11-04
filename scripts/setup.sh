#!/bin/bash
# HRNet Pose Detection - Linux/macOS Setup Script
# Automated setup for Unix-like systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_VENV=false
SKIP_TESTS=false
PYTHON_CMD="python3"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}🚀 HRNet Pose Detection - Linux/macOS Setup${NC}"
echo -e "${GREEN}===========================================${NC}"

# Check Python installation
echo -e "\n${YELLOW}📍 Checking Python installation...${NC}"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}❌ Python not found. Please install Python 3.8+ and try again.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"

# Check Python version
PYTHON_VERSION_NUM=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION_NUM" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.8+ required, found $PYTHON_VERSION_NUM${NC}"
    exit 1
fi

# Create virtual environment
if [ "$SKIP_VENV" = false ]; then
    echo -e "\n${YELLOW}📍 Creating virtual environment...${NC}"
    if [ -d "venv" ]; then
        echo -e "${YELLOW}⚠️  Virtual environment already exists. Removing...${NC}"
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    
    # Activate virtual environment
    echo -e "\n${YELLOW}📍 Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
    
    # Update Python command to use venv
    PYTHON_CMD="python"
fi

# Upgrade pip
echo -e "\n${YELLOW}📍 Upgrading pip...${NC}"
$PYTHON_CMD -m pip install --upgrade pip || echo -e "${YELLOW}⚠️  Pip upgrade failed, continuing...${NC}"

# Install dependencies
echo -e "\n${YELLOW}📍 Installing dependencies...${NC}"
$PYTHON_CMD -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Check model directory
echo -e "\n${YELLOW}📍 Checking model directory...${NC}"
if [ ! -d "model" ]; then
    echo -e "${YELLOW}📁 Creating model directory...${NC}"
    mkdir -p model
fi

if [ ! -f "model/model.onnx" ]; then
    echo -e "${YELLOW}⚠️  Model file not found at model/model.onnx${NC}"
    echo -e "${YELLOW}   Please download HRNet model from Qualcomm AI Hub${NC}"
    echo -e "${YELLOW}   and place it as model/model.onnx${NC}"
else
    MODEL_SIZE=$(du -h model/model.onnx | cut -f1)
    echo -e "${GREEN}✅ Model found: model/model.onnx ($MODEL_SIZE)${NC}"
fi

# Run tests
if [ "$SKIP_TESTS" = false ]; then
    echo -e "\n${YELLOW}📍 Running installation tests...${NC}"
    if $PYTHON_CMD main.py --test; then
        echo -e "${GREEN}✅ Tests passed!${NC}"
    else
        echo -e "${YELLOW}⚠️  Some tests failed, but installation may still work${NC}"
    fi
fi

# Check hardware capabilities
echo -e "\n${YELLOW}📍 Checking hardware capabilities...${NC}"
$PYTHON_CMD -c "
import sys
sys.path.insert(0, 'src')
try:
    from pose_detection.detectors.onnx_detector import ONNX_AVAILABLE
    from pose_detection.detectors.mediapipe_detector import MEDIAPIPE_AVAILABLE
    print('ONNX Runtime:', '✅ Available' if ONNX_AVAILABLE else '❌ Not available')
    print('MediaPipe:', '✅ Available' if MEDIAPIPE_AVAILABLE else '❌ Not available')
    
    if ONNX_AVAILABLE:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print('Available providers:', ', '.join(providers))
except Exception as e:
    print('Error checking capabilities:', e)
"

echo -e "\n${GREEN}🎉 Setup completed!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo -e "${CYAN}Next steps:${NC}"
echo -e "${WHITE}1. Ensure model file is at model/model.onnx${NC}"
echo -e "${WHITE}2. Generate context for faster startup: python main.py --generate-context${NC}"
echo -e "${WHITE}3. Test the application: python main.py --test${NC}"
echo -e "${WHITE}4. Start pose detection: python main.py${NC}"

if [ "$SKIP_VENV" = false ]; then
    echo -e "\n${CYAN}To activate environment in future sessions:${NC}"
    echo -e "${WHITE}source venv/bin/activate${NC}"
fi