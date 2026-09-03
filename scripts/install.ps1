[CmdletBinding()]
param(
    [string]$ApiKey,
    [string]$BaseUrl = "https://api.tmcra.com",
    [string]$AuthorizationUrl = "https://tmcra.com",
    [string]$ScopeNamespace = "tmcra",
    [string]$NodePath,
    [switch]$SkipConfigure,
    [switch]$SkipPluginInstall,
    [switch]$NoBrowser,
    [switch]$ProgressJson,
    [switch]$ApiOnlyCheck
)

$ErrorActionPreference = "Stop"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$pluginRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent (Split-Path -Parent $pluginRoot)

function Find-CodexCli {
    $candidate = Get-Command codex -ErrorAction SilentlyContinue
    if ($candidate -and $candidate.Source -and $candidate.Source -notlike "*WindowsApps*") {
        $path = $candidate.Source
    } else {
        $paths = @(
            Get-Item -LiteralPath "$HOME\.codex\plugins\.plugin-appserver\codex.exe" -ErrorAction SilentlyContinue
            Get-ChildItem -LiteralPath "$env:LOCALAPPDATA\OpenAI\Codex\bin" -Recurse -Filter codex.exe -ErrorAction SilentlyContinue
        ) | Where-Object { $_ -and $_.FullName } | Sort-Object LastWriteTime -Descending
        if ($paths.Count -eq 0) {
            throw "Codex CLI was not found. Install or update the Codex desktop app first."
        }
        $path = $paths[0].FullName
    }
    & $path plugin marketplace --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "This Codex version does not support plugins. Update the Codex desktop app first."
    }
    & $path plugin add --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "This Codex version does not support plugin installation. Update Codex first."
    }
    return $path
}

function Repair-CodexRuntime([string]$CodexPath, [ref]$BackupPathReference) {
    Write-TmcraProgress 'configure_codex' 'running' 'Enabling lifecycle hooks and removing legacy TMCRA MCP registrations.'
    $codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME '.codex' }
    $codexConfig = Join-Path $codexHome 'config.toml'
    $createdBackupPath = $null
    if (Test-Path -LiteralPath $codexConfig -PathType Leaf) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $createdBackupPath = "$codexConfig.tmcra-backup-$stamp"
        Copy-Item -LiteralPath $codexConfig -Destination $createdBackupPath -ErrorAction Stop
    }

    & $CodexPath features enable hooks *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'Codex could not enable lifecycle hooks.'
    }

    $servers = Invoke-CodexJson -CodexPath $CodexPath -Arguments @('mcp', 'list', '--json') -FailureMessage 'Could not inspect Codex MCP servers.'
    $legacyNames = @($servers) |
        Where-Object {
            $normalized = ([string]$_.name).ToLowerInvariant().Replace('_', '-')
            $normalized.StartsWith('tmcra-memory') -and ([string]$_.name) -ne 'tmcra-memory'
        } |
        ForEach-Object { [string]$_.name }
    foreach ($legacyName in $legacyNames) {
        & $CodexPath mcp remove $legacyName *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Could not remove legacy TMCRA MCP registration: $legacyName"
        }
    }

    Write-TmcraProgress 'configure_codex' 'completed' 'Codex hooks are enabled and exactly one TMCRA MCP registration is expected.'
    $BackupPathReference.Value = $createdBackupPath
}

function Test-NodeRuntime([string]$Path) {
    try {
        $versionText = & $Path --version 2>$null
        $exitCode = $LASTEXITCODE
    }
    catch {
        return $false
    }
    return (
        $exitCode -eq 0 -and
        ($versionText | Out-String).Trim() -match '^v(\d+)\.' -and
        [int]$Matches[1] -ge 18
    )
}

function Find-Node([string]$RequestedPath) {
    if ($RequestedPath) {
        $resolved = Resolve-Path -LiteralPath $RequestedPath -ErrorAction SilentlyContinue
        if (-not $resolved -or -not (Test-Path -LiteralPath $resolved.Path -PathType Leaf)) {
            throw "The explicit NodePath does not point to a readable executable file."
        }
        $path = $resolved.Path
        if (-not (Test-NodeRuntime $path)) {
            throw "The explicit NodePath must run Node.js 18 or newer. For Electron, set ELECTRON_RUN_AS_NODE=1 before starting this installer."
        }
        return $path
    }

    $candidates = @()
    $command = Get-Command node -ErrorAction SilentlyContinue
    if ($command -and $command.Source -and $command.Source -notlike "*WindowsApps*") {
        $candidates += $command.Source
    }
    $candidates += @(Get-ChildItem -LiteralPath "$env:LOCALAPPDATA\OpenAI\Codex\runtimes" -Recurse -Filter node.exe -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    ).FullName
    foreach ($path in @($candidates | Select-Object -Unique)) {
        if (Test-NodeRuntime $path) {
            return $path
        }
    }
    throw "Node.js 18 or newer was not found in PATH or the Codex runtime."
}

function Protect-TmcraConfig([string]$ConfigPath) {
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { return }
    & icacls $ConfigPath /inheritance:r /grant:r "${env:USERNAME}:(M)" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not restrict access to the TMCRA credential file." }
}

function Write-TmcraProgress([string]$Step, [string]$Status, [string]$Message) {
    if (-not $ProgressJson) { return }
    [ordered]@{
        event = "tmcra.install.progress"
        step = $Step
        status = $Status
        message = $Message
    } | ConvertTo-Json -Compress | Write-Output
}

function Invoke-CodexJson(
    [string]$CodexPath,
    [string[]]$Arguments,
    [string]$FailureMessage
) {
    $output = @(& $CodexPath @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $text = ($output | ForEach-Object { [string]$_ }) -join "`n"
    if ($exitCode -ne 0) {
        if (
            $text -match 'failed to (?:back up|remove existing) plugin cache entry' -and
            # Keep the script ASCII-safe for Windows PowerShell 5.1, which reads
            # UTF-8 files without a BOM using the active ANSI code page.
            $text -match '(?:os error 5|Access is denied|\u62D2\u7EDD\u8BBF\u95EE)'
        ) {
            throw "TMCRA_PLUGIN_CACHE_LOCKED"
        }
        throw $FailureMessage
    }
    try {
        return $text | ConvertFrom-Json
    }
    catch {
        throw $FailureMessage
    }
}

function Get-NormalizedPath([string]$Path) {
    if (-not $Path) { return "" }
    $Path = $Path -replace '^\\\\\?\\', ''
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Test-CachedPlugin(
    [string]$CachedPluginRoot,
    [object]$PluginManifest
) {
    $cachedManifestPath = Join-Path $CachedPluginRoot '.codex-plugin\plugin.json'
    if (-not (Test-Path -LiteralPath $cachedManifestPath -PathType Leaf)) {
        return $false
    }
    try {
        $cachedManifest = Get-Content -Raw -LiteralPath $cachedManifestPath | ConvertFrom-Json
    }
    catch {
        return $false
    }
    return (
        $cachedManifest.name -eq $PluginManifest.name -and
        $cachedManifest.version -eq $PluginManifest.version -and
        (Test-Path -LiteralPath (Join-Path $CachedPluginRoot '.mcp.json') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CachedPluginRoot 'hooks\hooks.json') -PathType Leaf)
    )
}

function Install-TmcraPlugin([string]$CodexPath) {
    $pluginManifest = Get-Content -Raw -LiteralPath (Join-Path $pluginRoot '.codex-plugin\plugin.json') | ConvertFrom-Json
    $expectedRoot = Get-NormalizedPath $repoRoot
    $codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME '.codex' }
    $cachedPluginRoot = Join-Path $codexHome "plugins\cache\tmcra-local\tmcra-memory\$($pluginManifest.version)"

    Write-TmcraProgress 'install_plugin' 'running' "Installing TMCRA Memory $($pluginManifest.version)."

    # A moved release directory makes `marketplace list` fail before it can
    # report the stale root. Removing the known local registration is safe and
    # does not uninstall the active plugin cache.
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        & $CodexPath plugin marketplace remove tmcra-local --json *> $null
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    $marketplaceAdd = Invoke-CodexJson -CodexPath $CodexPath -Arguments @('plugin', 'marketplace', 'add', $expectedRoot, '--json') -FailureMessage 'Could not register the TMCRA marketplace.'
    if ($marketplaceAdd.marketplaceName -ne 'tmcra-local') {
        throw "The extracted package does not declare the expected tmcra-local marketplace."
    }

    $installedBefore = Invoke-CodexJson -CodexPath $CodexPath -Arguments @('plugin', 'list', '--json') -FailureMessage 'Could not inspect installed Codex plugins.'
    $before = @($installedBefore.installed) |
        Where-Object { $_.pluginId -eq 'tmcra-memory@tmcra-local' } |
        Select-Object -First 1
    $beforeRoot = if ($before) { Get-NormalizedPath ([string]$before.marketplaceSource.source) } else { "" }
    $alreadyHealthy = (
        $before -and
        $before.version -eq $pluginManifest.version -and
        $beforeRoot.Equals($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Test-CachedPlugin $cachedPluginRoot $pluginManifest)
    )

    if (-not $alreadyHealthy) {
        # Never uninstall first. Codex can activate a new version beside a
        # Windows-locked old cache, but replacing the same live cache requires
        # renaming the whole directory and fails with os error 5.
        $pluginAdd = Invoke-CodexJson -CodexPath $CodexPath -Arguments @('plugin', 'add', 'tmcra-memory@tmcra-local', '--json') -FailureMessage 'Could not install the tmcra-memory Codex plugin.'
        if (
            $pluginAdd.pluginId -ne 'tmcra-memory@tmcra-local' -or
            $pluginAdd.version -ne $pluginManifest.version
        ) {
            throw "Codex cached a different TMCRA plugin version."
        }
        $cachedPluginRoot = Get-NormalizedPath ([string]$pluginAdd.installedPath)
    }

    $installedAfter = Invoke-CodexJson -CodexPath $CodexPath -Arguments @('plugin', 'list', '--json') -FailureMessage 'Could not verify the installed TMCRA plugin.'
    $after = @($installedAfter.installed) |
        Where-Object { $_.pluginId -eq 'tmcra-memory@tmcra-local' } |
        Select-Object -First 1
    if (-not $after -or $after.version -ne $pluginManifest.version) {
        throw "Codex did not activate the expected TMCRA plugin version."
    }
    $actualRoot = Get-NormalizedPath ([string]$after.marketplaceSource.source)
    if (-not $actualRoot.Equals($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Codex registered TMCRA from a different folder."
    }
    if (-not (Test-CachedPlugin $cachedPluginRoot $pluginManifest)) {
        throw "Codex did not create a complete TMCRA plugin cache."
    }
    Write-TmcraProgress 'install_plugin' 'completed' "TMCRA Memory $($pluginManifest.version) is installed."
}

try {
    $node = Find-Node $NodePath
    if ($ApiOnlyCheck -and -not $SkipPluginInstall) {
        throw "ApiOnlyCheck requires SkipPluginInstall and is only for isolated API tests."
    }
    $codex = if ($ApiOnlyCheck) { $null } else { Find-CodexCli }
    $configPath = if ($env:TMCRA_CONFIG_FILE) { $env:TMCRA_CONFIG_FILE } else { Join-Path $HOME ".config\tmcra\config.json" }

    if (-not $SkipPluginInstall) {
        Install-TmcraPlugin $codex
    }
    $codexBackup = $null
    if (-not $ApiOnlyCheck) {
        Repair-CodexRuntime $codex ([ref]$codexBackup)
    }

    if (-not $SkipConfigure) {
        # Older installers granted read/write without delete, which prevents the
        # atomic credential replacement used by device authorization.
        Protect-TmcraConfig $configPath
        if (-not $ApiKey -and $env:TMCRA_SETUP_API_KEY) {
            $ApiKey = $env:TMCRA_SETUP_API_KEY
        }
        if ($ApiKey) {
            $env:TMCRA_SETUP_API_KEY = $ApiKey
            $env:TMCRA_BASE_URL = $BaseUrl
            $env:TMCRA_SCOPE_NAMESPACE = $ScopeNamespace
            try {
                & $node (Join-Path $PSScriptRoot "configure.mjs") | Out-Null
                $configureExitCode = $LASTEXITCODE
            }
            finally {
                Remove-Item Env:TMCRA_SETUP_API_KEY -ErrorAction SilentlyContinue
                $ApiKey = $null
            }
            if ($configureExitCode -ne 0) { throw "TMCRA developer credential configuration failed." }
        }
        else {
            $env:TMCRA_AUTH_BASE_URL = $AuthorizationUrl
            if ($NoBrowser) { $env:TMCRA_DEVICE_NO_OPEN = "1" }
            try {
                $deviceLoginArguments = @((Join-Path $PSScriptRoot "device_login.mjs"))
                if ($ProgressJson) { $deviceLoginArguments += "--progress-json" }
                & $node @deviceLoginArguments
                $deviceLoginExitCode = $LASTEXITCODE
            }
            finally {
                Remove-Item Env:TMCRA_AUTH_BASE_URL -ErrorAction SilentlyContinue
                if ($NoBrowser) { Remove-Item Env:TMCRA_DEVICE_NO_OPEN -ErrorAction SilentlyContinue }
            }
            if ($deviceLoginExitCode -ne 0) { throw "TMCRA device authorization did not complete." }
        }
        Protect-TmcraConfig $configPath
    }

    if ($codex) { $env:TMCRA_CODEX_CLI = $codex }
    try {
        $checkArguments = @((Join-Path $PSScriptRoot "check_config.mjs"))
        if ($ApiOnlyCheck) { $checkArguments += "--api-only" }
        & $node @checkArguments
        $checkExitCode = $LASTEXITCODE
    }
    finally {
        Remove-Item Env:TMCRA_CODEX_CLI -ErrorAction SilentlyContinue
    }
    if ($checkExitCode -ne 0) { throw "TMCRA configuration check failed." }

    Write-Host "TMCRA Memory is installed and authorized in Codex Desktop. In Codex, run /hooks, trust all nine TMCRA lifecycle hooks, then start a new task and ask TMCRA to show status."
    if ($codexBackup) { Write-Host "Codex configuration backup: $codexBackup" }
}
catch {
    $code = 'installer_failed'
    $message = 'TMCRA setup did not complete.'
    $detail = [string]$_.Exception.Message
    $safeValidationMessages = @(
        'The explicit NodePath does not point to a readable executable file.',
        'The explicit NodePath must run Node.js 18 or newer. For Electron, set ELECTRON_RUN_AS_NODE=1 before starting this installer.',
        'Node.js 18 or newer was not found in PATH or the Codex runtime.',
        'Codex CLI was not found. Install or update the Codex desktop app first.',
        'This Codex version does not support plugins. Update the Codex desktop app first.',
        'This Codex version does not support plugin installation. Update Codex first.'
    )
    if ($safeValidationMessages -contains $detail) {
        $message = $detail
    }
    elseif ($detail -match 'TMCRA_PLUGIN_CACHE_LOCKED') {
        $code = 'plugin_cache_locked'
        $message = 'Codex is using the current plugin cache. Install the newer TMCRA package, then restart Codex once.'
    }
    if ($ProgressJson) {
        [ordered]@{
            event = 'tmcra.install.error'
            code = $code
            message = $message
        } | ConvertTo-Json -Compress | Write-Output
    }
    [Console]::Error.WriteLine($message)
    exit 1
}
