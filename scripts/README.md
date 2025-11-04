# Scripts Directory

This directory contains setup, installation, and utility scripts for different platforms.

## Scripts

### Setup Scripts
- `setup.ps1` - Windows PowerShell setup script
- `setup.sh` - Linux/macOS setup script
- `install_deps.py` - Cross-platform dependency installer

### Test Scripts
- `run_tests.ps1` - Windows test runner
- `run_tests.sh` - Linux/macOS test runner
- `validate_install.py` - Installation validator

### Utility Scripts
- `download_model.py` - Model downloader utility
- `generate_context.py` - Context generation utility
- `check_hardware.py` - Hardware compatibility checker

## Usage

### Windows
```powershell
# Setup environment
.\scripts\setup.ps1

# Run tests
.\scripts\run_tests.ps1

# Validate installation
python scripts\validate_install.py
```

### Linux/macOS
```bash
# Setup environment
./scripts/setup.sh

# Run tests
./scripts/run_tests.sh

# Validate installation
python scripts/validate_install.py
```

## Platform Requirements

- **Windows**: PowerShell 5.1+
- **Linux/macOS**: Bash 4.0+
- **Python**: 3.8+ (all scripts)