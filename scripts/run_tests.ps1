# HRNet Pose Detection - Windows Test Runner
# Run comprehensive tests on Windows

param(
    [switch]$Quick,
    [switch]$Coverage,
    [switch]$Lint
)

Write-Host "🧪 HRNet Pose Detection - Test Runner" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

# Activate virtual environment if exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "`n📍 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

if ($Quick) {
    Write-Host "`n⚡ Running quick tests..." -ForegroundColor Yellow
    & python main.py --test
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Quick tests passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Quick tests failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n🔍 Running comprehensive test suite..." -ForegroundColor Yellow
    & python tests\test_suite.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Comprehensive tests passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Some tests failed!" -ForegroundColor Red
    }
    
    Write-Host "`n🧪 Running unit tests..." -ForegroundColor Yellow
    & python -m pytest tests\ -v
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Unit tests passed!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some unit tests failed" -ForegroundColor Yellow
    }
}

if ($Coverage) {
    Write-Host "`n📊 Running coverage analysis..." -ForegroundColor Yellow
    & python -m pytest tests\ --cov=src --cov-report=html --cov-report=term
    Write-Host "📁 Coverage report generated in htmlcov/" -ForegroundColor Cyan
}

if ($Lint) {
    Write-Host "`n🔍 Running code linting..." -ForegroundColor Yellow
    & python -m flake8 src tests main.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Code linting passed!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Linting issues found" -ForegroundColor Yellow
    }
}

Write-Host "`n🎉 Test run completed!" -ForegroundColor Green