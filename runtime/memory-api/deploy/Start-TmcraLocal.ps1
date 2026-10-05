[CmdletBinding()]
param([string]$DataDir = (Join-Path $env:LOCALAPPDATA 'TMCRA\local'))
$ErrorActionPreference = 'Stop'
$apiRoot = Split-Path -Parent $PSScriptRoot
$localData = [IO.Path]::GetFullPath($DataDir)
$localPython = Join-Path $localData 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $localPython)) { throw 'Run Install-TmcraLocal.ps1 first.' }
$receipt = Get-Content -Raw -LiteralPath (Join-Path $localData 'installation.json') | ConvertFrom-Json
if ($receipt.api_root) { $apiRoot = $receipt.api_root }
$arguments = '-m tmcra_service.local_deployment run --root "{0}"' -f $localData
Start-Process -FilePath $localPython -ArgumentList $arguments -WorkingDirectory $apiRoot -WindowStyle Hidden -RedirectStandardError (Join-Path $localData 'launcher-error.log') -RedirectStandardOutput (Join-Path $localData 'launcher.log')
