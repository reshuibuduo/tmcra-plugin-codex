[CmdletBinding()]
param(
    [ValidateSet('lite-cpu','balanced-bge','quality-qwen')][string]$Profile = 'lite-cpu',
    [string]$DataDir = (Join-Path $env:LOCALAPPDATA 'TMCRA\local'),
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [switch]$PrepareOnly,
    [switch]$WaitReady
)
$ErrorActionPreference = 'Stop'
$apiRoot = Split-Path -Parent $PSScriptRoot
$localData = [IO.Path]::GetFullPath($DataDir)
if ([IO.Path]::GetPathRoot($localData) -eq $localData) { throw 'Choose a dedicated TMCRA data folder.' }
foreach ($broadPath in @($env:USERPROFILE,$env:LOCALAPPDATA,$env:APPDATA)) {
    if ($broadPath -and $localData.TrimEnd('\') -eq [IO.Path]::GetFullPath($broadPath).TrimEnd('\')) { throw 'Choose a dedicated TMCRA data folder.' }
}
if ($env:TMCRA_CONFIG_FILE) { throw 'An explicit TMCRA_CONFIG_FILE override is active. Clear this advanced override before choosing automatic local installation.' }
. (Join-Path $PSScriptRoot 'Local-SetupHelpers.ps1')
Protect-TmcraLocalPath $localData -Directory
$setupLock = [IO.File]::Open((Join-Path $localData 'setup.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
try {
    # The supervisor holds run.lock for its full lifetime. Do not update an active environment.
    $runProbe = [IO.File]::Open((Join-Path $localData 'run.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $runProbe.Dispose()
    Select-TmcraLocalProfile $localData $Profile
    $apiRoot = Copy-TmcraManagedRuntime $apiRoot $localData
    Remove-Item Env:TMCRA_DEPLOYMENT_MODE,Env:PYTHONPATH -ErrorAction SilentlyContinue
    Enable-TmcraDownloadProxy
    $localPython = Get-TmcraLocalPython $localData
$useCuda = $Device -eq 'cuda'
if ($Device -eq 'auto' -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
    $useCuda = $LASTEXITCODE -eq 0
}
if ($useCuda) {
    & $localPython -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
} else {
    & $localPython -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $localPython -m pip install -r (Join-Path $apiRoot 'requirements-tmcra-service.txt') 'transformers==4.57.6' 'huggingface-hub==0.36.2' 'sentencepiece==0.2.1' 'safetensors==0.8.0'
if ($LASTEXITCODE -ne 0) { throw 'Local dependency installation failed.' }
$env:PATH = (Split-Path -Parent $localPython) + [IO.Path]::PathSeparator + $env:PATH
Push-Location $apiRoot
try {
    $runtimeDevice = if ($useCuda) { 'cuda' } else { 'cpu' }
    & $localPython -m tmcra_service.local_deployment prepare --root $localData --profile $Profile --device $runtimeDevice --auto-ports
    if ($LASTEXITCODE -ne 0) { throw 'Local model preparation failed; retained downloads can be resumed.' }
    if (-not $PrepareOnly) {
        $arguments = '-m tmcra_service.local_deployment run --root "{0}"' -f $localData
        $service = Start-Process -FilePath $localPython -ArgumentList $arguments -WorkingDirectory $apiRoot -WindowStyle Hidden -RedirectStandardError (Join-Path $localData 'launcher-error.log') -RedirectStandardOutput (Join-Path $localData 'launcher.log') -PassThru
        Write-Output '{"event":"starting","message":"Local services are performing their full startup checks."}'
        if ($WaitReady) {
            $receipt = Get-Content -Raw -LiteralPath (Join-Path $localData 'installation.json') | ConvertFrom-Json
            $deadline = [DateTime]::UtcNow.AddMinutes(15)
            do {
                $service.Refresh()
                if ($service.HasExited) { throw 'Local service exited during startup; inspect launch-error.json. The local selection is retained.' }
                try {
                    $ready = Invoke-RestMethod -Uri "http://127.0.0.1:$($receipt.api_port)/readyz" -TimeoutSec 3
                    if ($ready.status -eq 'ready' -or $ready.ready -eq $true) { Write-Output '{"event":"ready","message":"Local memory is ready; no TMCRA account or server is required."}'; return }
                } catch { }
                Start-Sleep -Seconds 2
            } while ([DateTime]::UtcNow -lt $deadline)
            throw 'Startup did not reach ready in 15 minutes; inspect the retained local logs.'
        }
    }
} finally { Pop-Location }
} finally { $setupLock.Dispose() }
