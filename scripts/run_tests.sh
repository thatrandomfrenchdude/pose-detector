#!/bin/bash
# HRNet Pose Detection - Linux/macOS Test Runner
# Run comprehensive tests on Unix-like systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
QUICK=false
COVERAGE=false
LINT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quick)
            QUICK=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -l|--lint)
            LINT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-q|--quick] [-c|--coverage] [-l|--lint]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}🧪 HRNet Pose Detection - Test Runner${NC}"
echo -e "${GREEN}====================================${NC}"

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo -e "\n${YELLOW}📍 Activating virtual environment...${NC}"
    source venv/bin/activate
fi

if [ "$QUICK" = true ]; then
    echo -e "\n${YELLOW}⚡ Running quick tests...${NC}"
    if python main.py --test; then
        echo -e "${GREEN}✅ Quick tests passed!${NC}"
    else
        echo -e "${RED}❌ Quick tests failed!${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}🔍 Running comprehensive test suite...${NC}"
    if python tests/test_suite.py; then
        echo -e "${GREEN}✅ Comprehensive tests passed!${NC}"
    else
        echo -e "${RED}❌ Some tests failed!${NC}"
    fi
    
    echo -e "\n${YELLOW}🧪 Running unit tests...${NC}"
    if python -m pytest tests/ -v; then
        echo -e "${GREEN}✅ Unit tests passed!${NC}"
    else
        echo -e "${YELLOW}⚠️  Some unit tests failed${NC}"
    fi
fi

if [ "$COVERAGE" = true ]; then
    echo -e "\n${YELLOW}📊 Running coverage analysis...${NC}"
    python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
    echo -e "${CYAN}📁 Coverage report generated in htmlcov/${NC}"
fi

if [ "$LINT" = true ]; then
    echo -e "\n${YELLOW}🔍 Running code linting...${NC}"
    if python -m flake8 src tests main.py; then
        echo -e "${GREEN}✅ Code linting passed!${NC}"
    else
        echo -e "${YELLOW}⚠️  Linting issues found${NC}"
    fi
fi

echo -e "\n${GREEN}🎉 Test run completed!${NC}"