# Quick activation script for PowerShell
Set-Location $PSScriptRoot
& .\venv\Scripts\Activate.ps1
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Run: python test_local.py" -ForegroundColor Yellow
Write-Host "Or:  deactivate  (to exit venv)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
