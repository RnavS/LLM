param(
    [switch]$OpenBrowser = $true
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv312\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    py -3.12 -m venv .venv312
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements-local.txt

if ($OpenBrowser) {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 6
        Start-Process "http://127.0.0.1:8000"
    } | Out-Null
}

& $VenvPython -m uvicorn server.app:app --host 127.0.0.1 --port 8000
