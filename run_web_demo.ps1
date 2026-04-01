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
    $apiKey = [guid]::NewGuid().Guid
    @(
        "LLM_PRODUCT_NAME=MedBrief AI"
        "LLM_API_KEY=$apiKey"
        "LLM_HOST=127.0.0.1"
        "LLM_PORT=8000"
        "LLM_CHECKPOINT=checkpoints/advanced_local"
        "LLM_DEVICE=auto"
        "LLM_SYSTEM_PRESET=medbrief-medical"
        "LLM_KNOWLEDGE_INDEX=data/index/knowledge_index.pkl"
        "LLM_RETRIEVAL_TOP_K=4"
        "LLM_DISABLE_RETRIEVAL=false"
        "LLM_RESPONSE_MODE=assistant"
        "LLM_MODEL_BACKEND=auto"
        "LLM_OLLAMA_BASE_URL=http://127.0.0.1:11434"
        "LLM_OLLAMA_MODEL=llama3.1:8b"
        "LLM_OLLAMA_TIMEOUT_SECONDS=120.0"
        "LLM_OLLAMA_KEEP_ALIVE=10m"
        "LLM_WEB_SEARCH_ENABLED=true"
        "LLM_WEB_SEARCH_MAX_RESULTS=4"
        "LLM_WEB_TIMEOUT_SECONDS=3.0"
        "LLM_GENERATION_TIMEOUT_SECONDS=40"
        "LLM_SESSION_COOKIE_NAME=medbrief_session"
        "LLM_SESSION_COOKIE_MAX_AGE_DAYS=30"
        "LLM_SESSION_SECRET=local-dev-session-secret"
        "LLM_MESSAGE_RATE_LIMIT=20"
        "LLM_MESSAGE_RATE_WINDOW_SECONDS=60"
        "LLM_API_KEY_SELF_SERVE_ENABLED=false"
        "LLM_WEB_CACHE_DIR=data/web_cache"
        "LLM_DATABASE_PATH=data/app/medbrief.sqlite3"
        "LLM_FRONTEND_DIST=frontend_static"
        "LLM_ALLOW_ORIGINS=http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8090,http://localhost:8090"
    ) | Set-Content -Path ".env" -Encoding UTF8
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host "Launching MedBrief AI at http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "The website opens directly and stores session state in the browser cookie." -ForegroundColor DarkCyan
if ($OpenBrowser) {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 4
        Start-Process "http://127.0.0.1:8000"
    } | Out-Null
}

& $VenvPython -m uvicorn server.app:app --host 127.0.0.1 --port 8000
