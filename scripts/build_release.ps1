[CmdletBinding()]
param(
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$pluginRoot = Split-Path -Parent $PSScriptRoot
$defaultDownloadsRoot = Join-Path $pluginRoot "release"
if (-not $OutputPath) {
    $OutputPath = Join-Path $defaultDownloadsRoot "tmcra-codex-latest.zip"
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$downloadsRoot = [System.IO.Path]::GetDirectoryName($resolvedOutput)
if ([System.IO.Path]::GetFileName($resolvedOutput) -ne "tmcra-codex-latest.zip") {
    throw "OutputPath must end with tmcra-codex-latest.zip."
}

$pluginManifestPath = Join-Path $pluginRoot ".codex-plugin\plugin.json"
$pluginManifest = Get-Content -Raw -LiteralPath $pluginManifestPath | ConvertFrom-Json
$pluginVersion = [string]$pluginManifest.version
if ($pluginVersion -notmatch '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z.-]+)?$') {
    throw "Plugin version must be a release-orderable semantic version without build metadata."
}

$versionedOutput = Join-Path $downloadsRoot "tmcra-codex-$pluginVersion.zip"
$releaseManifestPath = Join-Path $downloadsRoot "tmcra-codex-release.json"
$versionedSha256Path = "$versionedOutput.sha256"
$aliasSha256Path = "$resolvedOutput.sha256"

$releaseFiles = @(
    [ordered]@{ Source = "packaging/.agents/plugins/marketplace.json"; Archive = ".agents/plugins/marketplace.json" },
    [ordered]@{ Source = ".codex-plugin/plugin.json"; Archive = "plugins/tmcra-memory/.codex-plugin/plugin.json" },
    [ordered]@{ Source = ".mcp.json"; Archive = "plugins/tmcra-memory/.mcp.json" },
    [ordered]@{ Source = "README.md"; Archive = "plugins/tmcra-memory/README.md" },
    [ordered]@{ Source = "hooks/hook_common.mjs"; Archive = "plugins/tmcra-memory/hooks/hook_common.mjs" },
    [ordered]@{ Source = "hooks/hooks.json"; Archive = "plugins/tmcra-memory/hooks/hooks.json" },
    [ordered]@{ Source = "hooks/post_compact.mjs"; Archive = "plugins/tmcra-memory/hooks/post_compact.mjs" },
    [ordered]@{ Source = "hooks/post_tool_use.mjs"; Archive = "plugins/tmcra-memory/hooks/post_tool_use.mjs" },
    [ordered]@{ Source = "hooks/pre_compact.mjs"; Archive = "plugins/tmcra-memory/hooks/pre_compact.mjs" },
    [ordered]@{ Source = "hooks/run_hook.mjs"; Archive = "plugins/tmcra-memory/hooks/run_hook.mjs" },
    [ordered]@{ Source = "hooks/session_start.mjs"; Archive = "plugins/tmcra-memory/hooks/session_start.mjs" },
    [ordered]@{ Source = "hooks/stop.mjs"; Archive = "plugins/tmcra-memory/hooks/stop.mjs" },
    [ordered]@{ Source = "hooks/subagent_start.mjs"; Archive = "plugins/tmcra-memory/hooks/subagent_start.mjs" },
    [ordered]@{ Source = "hooks/subagent_stop.mjs"; Archive = "plugins/tmcra-memory/hooks/subagent_stop.mjs" },
    [ordered]@{ Source = "hooks/user_prompt_submit.mjs"; Archive = "plugins/tmcra-memory/hooks/user_prompt_submit.mjs" },
    [ordered]@{ Source = "scripts/check_config.mjs"; Archive = "plugins/tmcra-memory/scripts/check_config.mjs" },
    [ordered]@{ Source = "scripts/configure.mjs"; Archive = "plugins/tmcra-memory/scripts/configure.mjs" },
    [ordered]@{ Source = "scripts/device_login.mjs"; Archive = "plugins/tmcra-memory/scripts/device_login.mjs" },
    [ordered]@{ Source = "scripts/drain_outbox.mjs"; Archive = "plugins/tmcra-memory/scripts/drain_outbox.mjs" },
    [ordered]@{ Source = "scripts/history_import.mjs"; Archive = "plugins/tmcra-memory/scripts/history_import.mjs" },
    [ordered]@{ Source = "scripts/install.ps1"; Archive = "plugins/tmcra-memory/scripts/install.ps1" },
    [ordered]@{ Source = "scripts/install.sh"; Archive = "plugins/tmcra-memory/scripts/install.sh" },
    [ordered]@{ Source = "scripts/mcp_server.mjs"; Archive = "plugins/tmcra-memory/scripts/mcp_server.mjs" },
    [ordered]@{ Source = "scripts/project_bootstrap.mjs"; Archive = "plugins/tmcra-memory/scripts/project_bootstrap.mjs" },
    [ordered]@{ Source = "scripts/project_init.mjs"; Archive = "plugins/tmcra-memory/scripts/project_init.mjs" },
    [ordered]@{ Source = "scripts/tmcra_client.mjs"; Archive = "plugins/tmcra-memory/scripts/tmcra_client.mjs" },
    [ordered]@{ Source = "skills/manage-tmcra-memory/agents/openai.yaml"; Archive = "plugins/tmcra-memory/skills/manage-tmcra-memory/agents/openai.yaml" },
    [ordered]@{ Source = "skills/manage-tmcra-memory/SKILL.md"; Archive = "plugins/tmcra-memory/skills/manage-tmcra-memory/SKILL.md" },
    [ordered]@{ Source = "packaging/INSTALL-TMCRA-CODEX.md"; Archive = "INSTALL-TMCRA-CODEX.md" },
    [ordered]@{ Source = "packaging/Install-TMCRA.ps1"; Archive = "Install-TMCRA.ps1" },
    [ordered]@{ Source = "packaging/install.sh"; Archive = "install.sh" }
)

foreach ($entry in $releaseFiles) {
    $sourcePath = Join-Path $pluginRoot ($entry.Source.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Release source is missing $($entry.Source)"
    }
}

New-Item -ItemType Directory -Force -Path $downloadsRoot | Out-Null
$temporaryArchive = Join-Path $downloadsRoot ".tmcra-codex-$([guid]::NewGuid().ToString('N')).zip"
$temporaryAlias = Join-Path $downloadsRoot ".tmcra-codex-alias-$([guid]::NewGuid().ToString('N')).zip"
$temporaryManifest = Join-Path $downloadsRoot ".tmcra-codex-release-$([guid]::NewGuid().ToString('N')).json"
$temporaryVersionedSha256 = Join-Path $downloadsRoot ".tmcra-codex-versioned-$([guid]::NewGuid().ToString('N')).sha256"
$temporaryAliasSha256 = Join-Path $downloadsRoot ".tmcra-codex-alias-$([guid]::NewGuid().ToString('N')).sha256"

function Write-Utf8NoBom([string]$Path, [string]$Value) {
    [System.IO.File]::WriteAllText($Path, $Value, (New-Object System.Text.UTF8Encoding($false)))
}

function Normalize-ZipCentralDirectory([string]$Path) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $minimumOffset = [Math]::Max(0, $bytes.Length - 65557)
    $endOffset = -1
    for ($index = $bytes.Length - 22; $index -ge $minimumOffset; $index--) {
        if (
            $bytes[$index] -eq 0x50 -and
            $bytes[$index + 1] -eq 0x4b -and
            $bytes[$index + 2] -eq 0x05 -and
            $bytes[$index + 3] -eq 0x06
        ) {
            $endOffset = $index
            break
        }
    }
    if ($endOffset -lt 0) {
        throw "ZIP end-of-central-directory record was not found."
    }

    $entryCount = [System.BitConverter]::ToUInt16($bytes, $endOffset + 10)
    $centralOffset = [int][System.BitConverter]::ToUInt32($bytes, $endOffset + 16)
    for ($entryIndex = 0; $entryIndex -lt $entryCount; $entryIndex++) {
        if (
            $bytes[$centralOffset] -ne 0x50 -or
            $bytes[$centralOffset + 1] -ne 0x4b -or
            $bytes[$centralOffset + 2] -ne 0x01 -or
            $bytes[$centralOffset + 3] -ne 0x02
        ) {
            throw "Invalid ZIP central-directory entry at offset $centralOffset."
        }

        # Canonical Unix creator metadata and regular-file mode 0644.
        $bytes[$centralOffset + 5] = 3
        $bytes[$centralOffset + 38] = 0
        $bytes[$centralOffset + 39] = 0
        $bytes[$centralOffset + 40] = 0xa4
        $bytes[$centralOffset + 41] = 0x81

        $nameLength = [System.BitConverter]::ToUInt16($bytes, $centralOffset + 28)
        $extraLength = [System.BitConverter]::ToUInt16($bytes, $centralOffset + 30)
        $commentLength = [System.BitConverter]::ToUInt16($bytes, $centralOffset + 32)
        $centralOffset += 46 + $nameLength + $extraLength + $commentLength
    }
    [System.IO.File]::WriteAllBytes($Path, $bytes)
}

function Publish-Atomic([string]$Source, [string]$Destination) {
    if (Test-Path -LiteralPath $Destination) {
        $backup = "$Destination.$([guid]::NewGuid().ToString('N')).bak"
        try {
            [System.IO.File]::Replace($Source, $Destination, $backup, $true)
        }
        finally {
            Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
        }
    }
    else {
        [System.IO.File]::Move($Source, $Destination)
    }
}

try {
    $runtimeFiles = @($releaseFiles | ForEach-Object { [string]$_.Archive })
    Add-Type -AssemblyName System.IO.Compression
    $archiveStream = [System.IO.File]::Open(
        $temporaryArchive,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    $createdArchive = [System.IO.Compression.ZipArchive]::new(
        $archiveStream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false
    )
    $releaseUtf8 = New-Object System.Text.UTF8Encoding($false)
    try {
        foreach ($entry in $releaseFiles) {
            $sourcePath = Join-Path $pluginRoot ($entry.Source.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
            $zipEntry = $createdArchive.CreateEntry(
                [string]$entry.Archive,
                [System.IO.Compression.CompressionLevel]::Optimal
            )
            $zipEntry.LastWriteTime = [System.DateTimeOffset]::new(
                2000,
                1,
                1,
                0,
                0,
                0,
                [System.TimeSpan]::Zero
            )
            $entryStream = $zipEntry.Open()
            try {
                # Every release entry is text. Canonical UTF-8/LF bytes make
                # archives reproducible across Windows and Unix checkouts.
                $content = [System.IO.File]::ReadAllText($sourcePath)
                $content = $content.Replace("`r`n", "`n").Replace("`r", "`n")
                $contentBytes = $releaseUtf8.GetBytes($content)
                $entryStream.Write($contentBytes, 0, $contentBytes.Length)
            }
            finally {
                $entryStream.Dispose()
            }
        }
    }
    finally {
        $createdArchive.Dispose()
        $archiveStream.Dispose()
    }

    Normalize-ZipCentralDirectory $temporaryArchive

    $inspectionStream = [System.IO.File]::Open(
        $temporaryArchive,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read
    )
    $archive = [System.IO.Compression.ZipArchive]::new(
        $inspectionStream,
        [System.IO.Compression.ZipArchiveMode]::Read,
        $false
    )
    try {
        $archiveFiles = @(
            $archive.Entries |
                Where-Object { -not [string]::IsNullOrEmpty($_.Name) } |
                ForEach-Object { $_.FullName.Replace("\", "/") }
        )
    }
    finally {
        $archive.Dispose()
        $inspectionStream.Dispose()
    }
    $unexpected = @($archiveFiles | Where-Object { $_ -notin $runtimeFiles })
    $missing = @($runtimeFiles | Where-Object { $_ -notin $archiveFiles })
    if ($unexpected.Count -gt 0) {
        throw "Release archive contains unexpected files: $($unexpected -join ', ')"
    }
    if ($missing.Count -gt 0) {
        throw "Release archive is missing files: $($missing -join ', ')"
    }

    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $temporaryArchive
    # PowerShell hides leading-dot files from Get-Item on Unix unless -Force is
    # supplied. FileInfo reads the archive consistently on every runner.
    $bytes = [System.IO.FileInfo]::new($temporaryArchive).Length
    Copy-Item -LiteralPath $temporaryArchive -Destination $temporaryAlias
    Write-Utf8NoBom $temporaryVersionedSha256 "$($hash.Hash.ToLowerInvariant())  $([System.IO.Path]::GetFileName($versionedOutput))`n"
    Write-Utf8NoBom $temporaryAliasSha256 "$($hash.Hash.ToLowerInvariant())  $([System.IO.Path]::GetFileName($resolvedOutput))`n"

    $releaseManifest = [ordered]@{
        schemaVersion = 1
        plugin = [ordered]@{
            name = [string]$pluginManifest.name
            version = $pluginVersion
        }
        archive = [ordered]@{
            latest = [System.IO.Path]::GetFileName($resolvedOutput)
            versioned = [System.IO.Path]::GetFileName($versionedOutput)
            latestSha256 = [System.IO.Path]::GetFileName($aliasSha256Path)
            versionedSha256 = [System.IO.Path]::GetFileName($versionedSha256Path)
            bytes = $bytes
            sha256 = $hash.Hash.ToLowerInvariant()
            entryCount = $archiveFiles.Count
        }
        install = [ordered]@{
            windows = ".\Install-TMCRA.ps1"
            macosLinux = "sh ./install.sh"
        }
        requirements = [ordered]@{
            node = ">=18"
            codexPluginCli = $true
        }
        generatedAtUtc = [DateTime]::UtcNow.ToString("o")
    }
    Write-Utf8NoBom $temporaryManifest "$(($releaseManifest | ConvertTo-Json -Depth 5))`n"

    Publish-Atomic $temporaryArchive $versionedOutput
    Publish-Atomic $temporaryAlias $resolvedOutput
    Publish-Atomic $temporaryVersionedSha256 $versionedSha256Path
    Publish-Atomic $temporaryAliasSha256 $aliasSha256Path
    Publish-Atomic $temporaryManifest $releaseManifestPath

    [pscustomobject]@{
        OutputPath = $resolvedOutput
        VersionedOutputPath = $versionedOutput
        AliasSha256Path = $aliasSha256Path
        VersionedSha256Path = $versionedSha256Path
        ReleaseManifestPath = $releaseManifestPath
        Version = $pluginVersion
        Bytes = $bytes
        Sha256 = $hash.Hash.ToLowerInvariant()
        EntryCount = $archiveFiles.Count
    }
}
finally {
    Remove-Item -LiteralPath $temporaryArchive -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $temporaryAlias -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $temporaryManifest -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $temporaryVersionedSha256 -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $temporaryAliasSha256 -Force -ErrorAction SilentlyContinue
}
