# HRNet Pose Detection - Windows Setup Script
# Automated setup for Windows systems

param(
    [switch]$SkipVenv,
    [switch]$SkipTests,
    [string]$PythonPath = "python"
)

Write-Host "🚀 HRNet Pose Detection - Windows Setup" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

# Check Python installation
Write-Host "`n📍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = & $PythonPath --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ and try again." -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-not $SkipVenv) {
    Write-Host "`n📍 Creating virtual environment..." -ForegroundColor Yellow
    if (Test-Path "venv") {
        Write-Host "⚠️  Virtual environment already exists. Removing..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "venv"
    }
    
    & $PythonPath -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
    
    # Activate virtual environment
    Write-Host "`n📍 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
}

# Upgrade pip
Write-Host "`n📍 Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Pip upgrade failed, continuing..." -ForegroundColor Yellow
}

# Install dependencies
Write-Host "`n📍 Installing dependencies..." -ForegroundColor Yellow
& pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Check model directory
Write-Host "`n📍 Checking model directory..." -ForegroundColor Yellow
if (-not (Test-Path "model")) {
    Write-Host "📁 Creating model directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "model" | Out-Null
}

if (-not (Test-Path "model\model.onnx")) {
    Write-Host "⚠️  Model file not found at model\model.onnx" -ForegroundColor Yellow
    Write-Host "   Please download HRNet model from Qualcomm AI Hub" -ForegroundColor Yellow
    Write-Host "   and place it as model\model.onnx" -ForegroundColor Yellow
} else {
    $modelSize = (Get-Item "model\model.onnx").Length / 1MB
    Write-Host "✅ Model found: model\model.onnx ($($modelSize.ToString('F1')) MB)" -ForegroundColor Green
}

# Run tests
if (-not $SkipTests) {
    Write-Host "`n📍 Running installation tests..." -ForegroundColor Yellow
    & python main.py --test
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Tests passed!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some tests failed, but installation may still work" -ForegroundColor Yellow
    }
}

# Check hardware capabilities
Write-Host "`n📍 Checking hardware capabilities..." -ForegroundColor Yellow
& python -c "
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
        print('QNN Provider:', '✅ Available' if 'QNNExecutionProvider' in providers else '❌ Not available')
except Exception as e:
    print('Error checking capabilities:', e)
"

Write-Host "`n🎉 Setup completed!" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Ensure model file is at model\model.onnx" -ForegroundColor White
Write-Host "2. Generate context for faster startup: python main.py --generate-context" -ForegroundColor White
Write-Host "3. Test the application: python main.py --test" -ForegroundColor White
Write-Host "4. Start pose detection: python main.py" -ForegroundColor White

if (-not $SkipVenv) {
    Write-Host "`nTo activate environment in future sessions:" -ForegroundColor Cyan
    Write-Host "venv\Scripts\Activate.ps1" -ForegroundColor White
}