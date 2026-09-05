function Protect-TmcraLocalPath([string]$Path, [switch]$Directory) {
    if ($Directory) { New-Item -ItemType Directory -Force -Path $Path | Out-Null }
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $rights = if ($Directory) { '(OI)(CI)F' } else { 'F' }
    & icacls.exe $Path /inheritance:r /grant:r "*${sid}:$rights" "*S-1-5-18:$rights" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not protect the local installation directory.' }
}

function Select-TmcraLocalProfile([string]$DataRoot, [string]$Profile) {
    if ($Profile -notin @('lite-cpu','balanced-bge','quality-qwen')) { throw 'Unknown local model profile.' }
    $selectionFile = if ($env:TMCRA_LOCAL_BINDING_FILE) { $env:TMCRA_LOCAL_BINDING_FILE } else { Join-Path $env:USERPROFILE '.config\tmcra\local-memory.json' }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $selectionFile) | Out-Null
    $temporarySelection = "$selectionFile.$([guid]::NewGuid().ToString('N')).tmp"
    $selection = @{schemaVersion=1;mode='local';dataRoot=[IO.Path]::GetFullPath($DataRoot);profile=$Profile}
    [IO.File]::WriteAllText($temporarySelection, ($selection | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Protect-TmcraLocalPath $temporarySelection
    if (Test-Path -LiteralPath $selectionFile) {
        # Windows PowerShell 5 converts a null backup argument to an invalid empty path.
        # Retain a private, non-secret selection backup and replace the marker atomically.
        $previousSelection = "$temporarySelection.previous"
        [IO.File]::Replace($temporarySelection,$selectionFile,$previousSelection)
        Protect-TmcraLocalPath $previousSelection
    }
    else { [IO.File]::Move($temporarySelection,$selectionFile) }
    Protect-TmcraLocalPath $selectionFile
}

function Copy-TmcraManagedRuntime([string]$Source, [string]$DataRoot) {
    $manifestPath = Join-Path $Source 'runtime-files.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Runtime inventory is missing; use the complete local installation package.' }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $identity = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant().Substring(0,24)
    $destination = Join-Path $DataRoot "service\$identity"
    $sourcePrefix = [IO.Path]::GetFullPath($Source).TrimEnd('\') + '\'
    $destinationPrefix = [IO.Path]::GetFullPath($destination).TrimEnd('\') + '\'
    foreach ($entry in $manifest.PSObject.Properties) {
        $inputPath = [IO.Path]::GetFullPath((Join-Path $Source $entry.Name))
        $outputPath = [IO.Path]::GetFullPath((Join-Path $destination $entry.Name))
        if (-not $inputPath.StartsWith($sourcePrefix,[StringComparison]::OrdinalIgnoreCase) -or
            -not $outputPath.StartsWith($destinationPrefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe runtime inventory path.' }
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Runtime integrity check failed: $($entry.Name)" }
        if (Test-Path -LiteralPath $outputPath) {
            if ((Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant() -ne $entry.Value) { throw 'Existing managed runtime was modified; inspect it before upgrading.' }
        } else {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null
            Copy-Item -LiteralPath $inputPath -Destination $outputPath
        }
    }
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $destination 'runtime-files.json')
    return $destination
}

function Enable-TmcraDownloadProxy {
    # Respect a user's Windows proxy for installation only. Runtime strips proxies.
    if ($env:HTTPS_PROXY) { return }
    $settings = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
    if ($settings.ProxyEnable -eq 1 -and $settings.ProxyServer -match '^(?:http://)?(127\.0\.0\.1:\d{1,5})$') {
        $env:HTTPS_PROXY = "http://$($Matches[1])"
        $env:HTTP_PROXY = $env:HTTPS_PROXY
        $env:NO_PROXY = '127.0.0.1,localhost,::1'
    }
}

function Get-TmcraLocalPython([string]$DataRoot) {
    $localPython = Join-Path $DataRoot 'venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) {
        & $localPython -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13) and sys.maxsize > 2**32'
        if ($LASTEXITCODE -ne 0) { throw 'The existing local Python is incompatible; retain its data and inspect the environment.' }
        return $localPython
    }
    $uvRoot = Join-Path $DataRoot 'runtime\uv-0.12.10'
    New-Item -ItemType Directory -Force -Path $uvRoot | Out-Null
    $archive = Join-Path $uvRoot 'uv.zip'
    $expected = 'f65744f94072152b1f86ba2aace4d01f1124d9a8ecb235805039e3718c36cac2'
    if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -ne $expected) {
        Write-Output '{"event":"installing_python","message":"Downloading the pinned Python environment manager."}' | Write-Host
        $partial = "$archive.partial"
        $download = @{UseBasicParsing=$true;Uri='https://github.com/astral-sh/uv/releases/download/0.12.10/uv-x86_64-pc-windows-msvc.zip';OutFile=$partial}
        if ($env:HTTPS_PROXY) { $download.Proxy = $env:HTTPS_PROXY }
        Invoke-WebRequest @download
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash.ToLowerInvariant() -ne $expected) { throw 'Python bootstrap checksum mismatch.' }
        Move-Item -LiteralPath $partial -Destination $archive -Force
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $uvRoot -Force
    $uv = Join-Path $uvRoot 'uv.exe'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $DataRoot 'runtime\python'
    $env:UV_CACHE_DIR = Join-Path $DataRoot 'cache\uv'
    & $uv venv --python 3.12 --managed-python --seed (Join-Path $DataRoot 'venv') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Automatic Python installation failed; downloaded files are retained for retry.' }
    return $localPython
}
