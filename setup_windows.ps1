$ErrorActionPreference = "Stop"
Write-Host "CropCare AI setup helper" -ForegroundColor Green
Write-Host "This installs dependencies and prepares the dataset/model folders."
if (-not (Test-Path ".\venv\Scripts\python.exe")) { py -3.11 -m venv venv }
$python = ".\venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path ".\dataset\PlantVillage" | Out-Null
New-Item -ItemType Directory -Force -Path ".\model" | Out-Null
Write-Host ""
Write-Host "Next: download a plant disease dataset you are permitted to use and extract its class folders into .\dataset\PlantVillage\"
Write-Host "Then run: .\venv\Scripts\python.exe model\train.py"
Write-Host "Finally run: .\venv\Scripts\python.exe app.py"
